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
def ReleaseService(self, service_ref, config_changes, release_track, tracker=None, asyn=False, allow_unauthenticated=None, for_replace=False, prefetch=False, build_image=None, build_pack=None, build_source=None, repo_to_create=None, already_activated_services=False, dry_run=False, generate_name=False, delegate_builds=False, base_image=None):
    """Change the given service in prod using the given config_changes.

    Ensures a new revision is always created, even if the spec of the revision
    has not changed.

    Args:
      service_ref: Resource, the service to release.
      config_changes: list, objects that implement Adjust().
      release_track: ReleaseTrack, the release track of a command calling this.
      tracker: StagedProgressTracker, to report on the progress of releasing.
      asyn: bool, if True, return without waiting for the service to be updated.
      allow_unauthenticated: bool, True if creating a hosted Cloud Run service
        which should also have its IAM policy set to allow unauthenticated
        access. False if removing the IAM policy to allow unauthenticated access
        from a service.
      for_replace: bool, If the change is for a replacing the service from a
        YAML specification.
      prefetch: the service, pre-fetched for ReleaseService. `False` indicates
        the caller did not perform a prefetch; `None` indicates a nonexistent
        service.
      build_image: The build image reference to the build.
      build_pack: The build pack reference to the build.
      build_source: The build source reference to the build.
      repo_to_create: Optional
        googlecloudsdk.command_lib.artifacts.docker_util.DockerRepo defining a
        repository to be created.
      already_activated_services: bool. If true, skip activation prompts for
        services
      dry_run: bool. If true, only validate the configuration.
      generate_name: bool. If true, create a revision name, otherwise add nonce.
      delegate_builds: bool. If true, use the Build API to submit builds.
      base_image: The build base image to opt-in automatic build image updates.

    Returns:
      service.Service, the service as returned by the server on the POST/PUT
       request to create/update the service.
    """
    if tracker is None:
        tracker = progress_tracker.NoOpStagedProgressTracker(stages.ServiceStages(allow_unauthenticated is not None, include_build=build_source is not None, include_create_repo=repo_to_create is not None), interruptable=True, aborted_message='aborted')
    if repo_to_create:
        self._CreateRepository(tracker, repo_to_create, skip_activation_prompt=already_activated_services)
    if build_source is not None:
        image_digest = deployer.CreateImage(tracker, build_image, build_source, build_pack, release_track, already_activated_services, self._region, service_ref, delegate_builds, base_image)
        if image_digest is None:
            return
        config_changes.append(_AddDigestToImageChange(image_digest))
    if prefetch is None:
        serv = None
    elif build_source:
        serv = self.GetService(service_ref)
    else:
        serv = prefetch or self.GetService(service_ref)
    if for_replace:
        with_image = True
    else:
        with_image = any((isinstance(c, config_changes_mod.ImageChange) for c in config_changes))
        if config_changes_mod.AdjustsTemplate(config_changes):
            if generate_name:
                self._AddRevisionForcingChange(serv, config_changes)
            else:
                config_changes.append(_NewRevisionNonceChange(_Nonce()))
            if serv and (not with_image):
                self._EnsureImageDigest(serv, config_changes)
    if serv and serv.metadata.deletionTimestamp is not None:
        raise serverless_exceptions.DeploymentFailedError('Service [{}] is in the process of being deleted.'.format(service_ref.servicesId))
    updated_service = self._UpdateOrCreateService(service_ref, config_changes, with_image, serv, dry_run)
    if allow_unauthenticated is not None:
        try:
            tracker.StartStage(stages.SERVICE_IAM_POLICY_SET)
            tracker.UpdateStage(stages.SERVICE_IAM_POLICY_SET, '')
            self.AddOrRemoveIamPolicyBinding(service_ref, allow_unauthenticated, ALLOW_UNAUTH_POLICY_BINDING_MEMBER, ALLOW_UNAUTH_POLICY_BINDING_ROLE)
            tracker.CompleteStage(stages.SERVICE_IAM_POLICY_SET)
        except api_exceptions.HttpError:
            warning_message = 'Setting IAM policy failed, try "gcloud beta run services {}-iam-policy-binding --region={region} --member=allUsers --role=roles/run.invoker {service}"'.format('add' if allow_unauthenticated else 'remove', region=self._region, service=service_ref.servicesId)
            tracker.CompleteStageWithWarning(stages.SERVICE_IAM_POLICY_SET, warning_message=warning_message)
    if not asyn and (not dry_run):
        if updated_service.conditions.IsReady():
            return updated_service
        getter = functools.partial(self.GetService, service_ref) if updated_service.operation_id is None else functools.partial(self.WaitService, updated_service.operation_id)
        poller = op_pollers.ServiceConditionPoller(getter, tracker, dependencies=stages.ServiceDependencies(), serv=updated_service)
        self.WaitForCondition(poller)
        for msg in run_condition.GetNonTerminalMessages(poller.GetConditions()):
            tracker.AddWarning(msg)
        updated_service = poller.GetResource()
    return updated_service