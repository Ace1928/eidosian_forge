from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from concurrent import futures
import encodings.idna  # pylint: disable=unused-import
import json
import mimetypes
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib import artifacts
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import remote_repo_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import upgrade_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import edit
from googlecloudsdk.core.util import parallel
import requests
def MigrateToArtifactRegistry(unused_ref, args):
    """Runs the automigrate wizard for the current project."""
    if args.projects:
        projects = args.projects.split(',')
    else:
        projects = [args.project or properties.VALUES.core.project.GetOrFail()]
    recent_images = args.recent_images
    last_uploaded_versions = args.last_uploaded_versions
    from_gcr = args.from_gcr
    to_pkg_dev = args.to_pkg_dev
    copy_only = args.copy_only
    canary_reads = args.canary_reads
    skip_iam = args.skip_iam_update
    if recent_images is not None and (recent_images < 30 or recent_images > 90):
        log.status.Print('--recent-images must be between 30 and 90 inclusive')
        return None
    if canary_reads is not None and (canary_reads < 1 or canary_reads > 100):
        log.status.Print('--canary-reads must be between 1 and 100 inclusive')
        return None
    if args.projects and (from_gcr or to_pkg_dev):
        log.status.Print('Projects argument may not be used when providing --from-gcr and --to-pkg-dev')
        return None
    if bool(from_gcr) != bool(to_pkg_dev):
        log.status.Print('--from-gcr and --to-pkg-dev-repo should be provided together')
        return None
    if last_uploaded_versions and recent_images:
        log.status.Print('Only one of --last-uploaded-versions and --recent-images can be used')
        return None
    if to_pkg_dev:
        s = from_gcr.split('/', 1)
        if len(s) != 2:
            log.status.Print('--from-gcr must be of the form {host}/{project}')
        gcr_host, gcr_project = s
        s = to_pkg_dev.split('/', 1)
        if len(s) != 2:
            log.status.Print('--to-pkg-dev must be of the form {project}/{repo}')
        ar_project, ar_repo = s
        if 'gcr.io' in ar_repo:
            log.status.Print('--to-pkg-dev is only used for pkg.dev repos. Use --projects to migrate to a gcr.io repo')
            return None
        if gcr_host not in _ALLOWED_GCR_REPO_LOCATION.keys():
            log.status.Print('{gcr_host} is not a valid gcr host. Valid hosts: {hosts}'.format(gcr_host=gcr_host, hosts=', '.join(_ALLOWED_GCR_REPO_LOCATION.keys())))
            return None
        location = _ALLOWED_GCR_REPO_LOCATION[gcr_host]
        host = '{}{}-docker.pkg.dev'.format(properties.VALUES.artifacts.registry_endpoint_prefix.Get(), location)
        if not copy_only:
            has_bucket = GetGCRRepos({k: v for k, v in _GCR_BUCKETS.items() if v['repository'] == gcr_host}, gcr_project)
            if not skip_iam:
                SetupAuthForRepository(gcr_project=gcr_project, ar_project=ar_project, host=gcr_host, repo={'location': location, 'repository': ar_repo}, has_bucket=has_bucket, pkg_dev=True)
        if not WrappedCopyImagesFromGCR([host], to_pkg_dev, recent_images, last_uploaded=last_uploaded_versions, copy_from=from_gcr, max_threads=args.max_threads):
            return None
        if not copy_only:
            log.status.Print('\nAny reference to {gcr} will still need to be updated to reference {ar}'.format(gcr=from_gcr, ar=host + '/' + to_pkg_dev))
        return None
    messages = ar_requests.GetMessages()
    if copy_only:
        copying_projects = projects
        enabled_projects = []
        disabled_projects = []
        invalid_projects = []
        partial_projects = []
    else:
        if not CheckRedirectionPermission(projects):
            return None
        redirection_state = GetRedirectionStates(projects)
        enabled_projects = []
        disabled_projects = []
        copying_projects = []
        invalid_projects = []
        partial_projects = []
        for project, state in redirection_state.items():
            if state == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED:
                enabled_projects.append(project)
            elif state == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED_AND_COPYING:
                copying_projects.append(project)
            elif state == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_DISABLED:
                disabled_projects.append(project)
            elif state == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_PARTIAL_AND_COPYING:
                partial_projects.append(project)
            else:
                invalid_projects.append(project)
    if invalid_projects:
        log.status.Print('Skipping migration for projects in unsppoted state: {}'.format(invalid_projects))
        if len(invalid_projects) == len(projects):
            return None
    if len(enabled_projects) == len(projects):
        log.status.Print('Artifact Registry is already handling all requests for *gcr.io repos for the provided projects. If there are images you still need to copy, use the --copy-only flag.')
        return None
    if enabled_projects:
        log.status.Print('Skipping already migrated projects: {}\n'.format(enabled_projects))
    if disabled_projects:
        if not MaybeCreateMissingRepos(disabled_projects, automigrate=True, dry_run=False):
            return None
    existing_repos = {}
    for project in disabled_projects + copying_projects + partial_projects:
        existing_repos[project] = GetExistingRepos(project)
    projects_to_redirect = []
    dangerous_projects = []
    repo_bucket_map = {}
    for project in disabled_projects:
        if not existing_repos[project]:
            log.status.Print('Skipping project {} because it has no Artifact Registry repos to migrate to'.format(project))
        missing_bucket_repos = []
        repos_with_gcr_buckets = GetGCRRepos(_GCR_BUCKETS, project)
        repo_bucket_map[project] = repos_with_gcr_buckets
        for g in repos_with_gcr_buckets:
            if g not in [r['repository'] for r in existing_repos[project]]:
                missing_bucket_repos.append(g)
        if missing_bucket_repos:
            dangerous_projects.append(project)
        else:
            projects_to_redirect.append(project)
    if projects_to_redirect or partial_projects:
        for project in dangerous_projects:
            log.status.Print('Skipping project {} because it has a Container Registry bucket without a corresponding Artifact Registry repository.'.format(project))
    elif dangerous_projects:
        c = console_attr.GetConsoleAttr()
        cont = console_io.PromptContinue('\n{project_str} has Container Registry buckets without corresponding Artifact Registry repositories. Existing Container Registry data will become innacessible.'.format(project_str='This project' if len(dangerous_projects) == 1 else 'Each project'), 'Do you wish to continue ' + c.Colorize('(not recommended)', 'red'), default=False)
        if not cont:
            return None
        projects_to_redirect = dangerous_projects
    if not copy_only and projects_to_redirect:
        pre_copied_projects = []
        log.status.Print('\nCopying initial images (additional images will be copied later)...\n')
        for project in projects_to_redirect:
            gcr_hosts = [r['repository'] for r in existing_repos[project]]
            last_uploaded_for_precopy = 100
            if last_uploaded_versions:
                last_uploaded_for_precopy = min(last_uploaded_versions, last_uploaded_for_precopy)
            if WrappedCopyImagesFromGCR(gcr_hosts, project, recent_images=7, last_uploaded=last_uploaded_for_precopy, convert_to_pkg_dev=True, max_threads=args.max_threads, pre_copy=True):
                pre_copied_projects.append(project)
        projects_to_redirect = pre_copied_projects
    if not skip_iam:
        for project in projects_to_redirect:
            continue_checking_auth = SetupAuthForProject(project, existing_repos[project], repo_bucket_map[project])
            if not continue_checking_auth:
                break
    projects_to_redirect.extend(partial_projects)
    if canary_reads and projects_to_redirect:
        log.status.Print(f'\nThe next step will redirect {canary_reads}% of *gcr.io read traffic to Artifact Registry. All pushes will still write to Container Registry. While canarying, Artifact Registry will attempt to copy missing images from Container Registry at request time.\n')
        update = console_io.PromptContinue('Projects to redirect: {}'.format(projects_to_redirect), default=False)
        if not update:
            return None
        for project in projects_to_redirect:
            if SetRedirectionStatus(project, messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_PARTIAL_AND_COPYING, pull_percent=canary_reads):
                copying_projects.append(project)
                log.status.Print(f'{canary_reads}% of *gcr.io read traffic is now being served by Artifact Registry for {project}. Missing images are copied from Container Registry.\nTo send traffic back to Container Registry, run:\n  gcloud artifacts settings disable-upgrade-redirection --project={project}\nTo send all traffic to Artifact Registry, re-run this script without --canary-reads')
                return None
    if projects_to_redirect:
        caveat = ''
        if recent_images:
            caveat = f' that have been pulled or pushed in the last {recent_images} days'
        log.status.Print(f"\nThe next step will redirect all *gcr.io traffic to Artifact Registry. Remaining Container Registry images{caveat} will be copied. During migration, Artifact Registry will serve *gcr.io requests for images it doesn't have yet by copying them from Container Registry at request time. Deleting images from *gcr.io repos in the middle of migration might not be effective.\n\nIMPORTANT: Make sure to update any relevant VPC-SC policies before migrating. Once *gcr.io is redirected to Artifact Registry, the artifactregistry.googleapis.com service will be checked for VPC-SC instead of containerregistry.googleapis.com.\n")
        update = console_io.PromptContinue('Projects to redirect: {}'.format(projects_to_redirect), default=True)
        if not update:
            return None
    for project in projects_to_redirect:
        if SetRedirectionStatus(project, messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED_AND_COPYING):
            copying_projects.append(project)
            log.status.Print('*gcr.io traffic is now being served by Artifact Registry for {project}. Missing images are being copied from Container Registry\nTo send traffic back to Container Registry, run:\n  gcloud artifacts settings disable-upgrade-redirection --project={project}\n'.format(project=project))
    if not copying_projects:
        return None
    if copy_only:
        log.status.Print('\nCopying images...\n')
    else:
        log.status.Print('\nCopying remaining images...\n')
    failed_copies = []
    to_enable = []
    unredirected_copying_projects = set()
    if copy_only:
        for project, state in GetRedirectionStates(projects).items():
            if state == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_DISABLED:
                unredirected_copying_projects.add(project)
    for project in copying_projects:
        gcr_hosts = [r['host'] for r in existing_repos[project]]
        convert_to_pkg_dev = project in unredirected_copying_projects
        if convert_to_pkg_dev:
            gcr_hosts = [r['repository'] for r in existing_repos[project]]
        if WrappedCopyImagesFromGCR(gcr_hosts, project, recent_images, last_uploaded=last_uploaded_versions, convert_to_pkg_dev=convert_to_pkg_dev, max_threads=args.max_threads):
            to_enable.append(project)
        else:
            failed_copies.append(project)
    if copy_only:
        return None
    if failed_copies:
        if to_enable:
            log.status.Print('\nOnly completing migration for successful projects')
        else:
            cont = console_io.PromptContinue('\nAll projects had image copy failures. Continuing will disable further copying and images will be missing.', 'Continue anyway?', default=False)
            if cont:
                to_enable = failed_copies
            if not cont:
                return None
    log.status.Print()
    for project in to_enable:
        if SetRedirectionStatus(project, messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_ENABLED):
            log.status.Print('*gcr.io traffic is now being fully served by Artifact Registry for {project}. Images will no longer be copied from Container Registry for this project.'.format(project=project))
            enabled_projects.append(project)
    log.status.Print('\nThe following projects are fully migrated: {}'.format(enabled_projects))
    remaining_projects = list(set(projects) - set(enabled_projects))
    if remaining_projects:
        log.status.Print('The following projects still need to finish being migrated: {}'.format(remaining_projects))
        log.status.Print("\nThis script can be re-run to migrate any projects that haven'tfinished.")