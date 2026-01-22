from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import promote_util
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Redeploy(base.CreateCommand):
    """Redeploy the last release to a target.

  Redeploy the last rollout that has a state of SUCCESSFUL or FAILED to a
  target.
  If rollout-id is not specified, a rollout ID will be generated.
  """
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddTargetResourceArg(parser, positional=True)
        flags.AddRolloutID(parser)
        flags.AddDeliveryPipeline(parser)
        flags.AddDescriptionFlag(parser)
        flags.AddAnnotationsFlag(parser, _ROLLOUT)
        flags.AddLabelsFlag(parser, _ROLLOUT)
        flags.AddStartingPhaseId(parser)
        flags.AddOverrideDeployPolicies(parser)

    @gcloud_exception.CatchHTTPErrorRaiseHTTPException(deploy_exceptions.HTTP_ERROR_FORMAT)
    def Run(self, args):
        target_ref = args.CONCEPTS.target.Parse()
        target_util.GetTarget(target_ref)
        ref_dict = target_ref.AsDict()
        pipeline_ref = resources.REGISTRY.Parse(args.delivery_pipeline, collection='clouddeploy.projects.locations.deliveryPipelines', params={'projectsId': ref_dict['projectsId'], 'locationsId': ref_dict['locationsId'], 'deliveryPipelinesId': args.delivery_pipeline})
        pipeline_obj = delivery_pipeline_util.GetPipeline(pipeline_ref.RelativeName())
        failed_redeploy_prefix = 'Cannot perform redeploy.'
        delivery_pipeline_util.ThrowIfPipelineSuspended(pipeline_obj, failed_redeploy_prefix)
        current_release_ref = _GetCurrentRelease(pipeline_ref, target_ref, rollout_util.ROLLOUT_IN_TARGET_FILTER_TEMPLATE)
        release_obj = release.ReleaseClient().Get(current_release_ref.RelativeName())
        if release_obj.abandoned:
            raise deploy_exceptions.AbandonedReleaseError(failed_redeploy_prefix, current_release_ref.RelativeName())
        messages = core_apis.GetMessagesModule('clouddeploy', 'v1')
        skaffold_support_state = release_util.GetSkaffoldSupportState(release_obj)
        skaffold_support_state_enum = messages.SkaffoldSupportedCondition.SkaffoldSupportStateValueValuesEnum
        if skaffold_support_state == skaffold_support_state_enum.SKAFFOLD_SUPPORT_STATE_MAINTENANCE_MODE:
            log.status.Print("WARNING: This release's Skaffold version is in maintenance mode and will be unsupported soon.\n https://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")
        if skaffold_support_state == skaffold_support_state_enum.SKAFFOLD_SUPPORT_STATE_UNSUPPORTED:
            raise core_exceptions.Error("You can't redeploy this release because the Skaffold version that was used to create the release is no longer supported.\nhttps://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")
        prompt = 'Are you sure you want to redeploy release {} to target {}?'.format(current_release_ref.Name(), target_ref.Name())
        console_io.PromptContinue(prompt, cancel_on_no=True)
        policies = deploy_policy_util.CreateDeployPolicyNamesFromIDs(pipeline_ref, args.override_deploy_policies)
        promote_util.Promote(current_release_ref, release_obj, target_ref.Name(), False, rollout_id=args.rollout_id, annotations=args.annotations, labels=args.labels, description=args.description, starting_phase_id=args.starting_phase_id, override_deploy_policies=policies)