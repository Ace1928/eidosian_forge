from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def DeployJob(self, job_ref, config_changes, release_track, tracker=None, asyn=False, build_image=None, build_pack=None, build_source=None, repo_to_create=None, prefetch=None, already_activated_services=False):
    """Deploy to create a new Cloud Run Job or to update an existing one.

    Args:
      job_ref: Resource, the job to create or update.
      config_changes: list, objects that implement Adjust().
      release_track: ReleaseTrack, the release track of a command calling this.
      tracker: StagedProgressTracker, to report on the progress of releasing.
      asyn: bool, if True, return without waiting for the job to be updated.
      build_image: The build image reference to the build.
      build_pack: The build pack reference to the build.
      build_source: The build source reference to the build.
      repo_to_create: Optional
        googlecloudsdk.command_lib.artifacts.docker_util.DockerRepo defining a
        repository to be created.
      prefetch: the job, pre-fetched for DeployJob. `None` indicates a
        nonexistent job so the job has to be created, else this is for an
        update.
      already_activated_services: bool. If true, skip activation prompts for
        services

    Returns:
      A job.Job object.
    """
    if tracker is None:
        tracker = progress_tracker.NoOpStagedProgressTracker(stages.JobStages(include_build=build_source is not None, include_create_repo=repo_to_create is not None), interruptable=True, aborted_message='aborted')
    if repo_to_create:
        self._CreateRepository(tracker, repo_to_create, skip_activation_prompt=already_activated_services)
    if build_source is not None:
        image_digest = deployer.CreateImage(tracker, build_image, build_source, build_pack, release_track, already_activated_services, self._region, job_ref)
        if image_digest is None:
            return
        config_changes.append(_AddDigestToImageChange(image_digest))
    is_create = not prefetch
    if is_create:
        return self.CreateJob(job_ref, config_changes, tracker, asyn)
    else:
        return self.UpdateJob(job_ref, config_changes, tracker, asyn)