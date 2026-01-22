from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Advance(base.CreateCommand):
    """Advances a rollout."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddRolloutResourceArg(parser, positional=True)
        flags.AddPhaseId(parser, required=False)
        flags.AddOverrideDeployPolicies(parser)

    @gcloud_exception.CatchHTTPErrorRaiseHTTPException(deploy_exceptions.HTTP_ERROR_FORMAT)
    def Run(self, args):
        rollout_ref = args.CONCEPTS.rollout.Parse()
        pipeline_ref = rollout_ref.Parent().Parent()
        pipeline_obj = delivery_pipeline_util.GetPipeline(pipeline_ref.RelativeName())
        failed_activity_msg = 'Cannot advance rollout {}.'.format(rollout_ref.RelativeName())
        delivery_pipeline_util.ThrowIfPipelineSuspended(pipeline_obj, failed_activity_msg)
        phase_id = args.phase_id
        if phase_id is None:
            phase_id = self._DetermineNextPhase(rollout_ref)
        log.status.Print('Advancing rollout {} to phase {}.\n'.format(rollout_ref.RelativeName(), phase_id))
        policies = deploy_policy_util.CreateDeployPolicyNamesFromIDs(pipeline_ref, args.override_deploy_policies)
        return rollout.RolloutClient().AdvanceRollout(rollout_ref.RelativeName(), phase_id, override_deploy_policies=policies)

    def _DetermineNextPhase(self, rollout_ref):
        """Determines the phase in which the advance command should apply."""
        messages = core_apis.GetMessagesModule('clouddeploy', 'v1')
        ro = rollout.RolloutClient().Get(rollout_ref.RelativeName())
        if ro.state != messages.Rollout.StateValueValuesEnum.IN_PROGRESS:
            raise deploy_exceptions.RolloutNotInProgressError(rollout_name=rollout_ref.RelativeName())
        pending_phase_index = None
        for index, phase in enumerate(ro.phases):
            if phase.state == messages.Phase.StateValueValuesEnum.PENDING:
                pending_phase_index = index
                break
        if pending_phase_index is None:
            raise deploy_exceptions.RolloutCannotAdvanceError(rollout_name=rollout_ref.RelativeName(), failed_activity_msg='No pending phases.')
        if pending_phase_index == 0:
            raise deploy_exceptions.RolloutCannotAdvanceError(rollout_name=rollout_ref.RelativeName(), failed_activity_msg='Cannot advance first phase of a rollout.')
        advanceable_prior_phase_states = [messages.Phase.StateValueValuesEnum.SUCCEEDED, messages.Phase.StateValueValuesEnum.SKIPPED]
        prior_phase = ro.phases[pending_phase_index - 1]
        if prior_phase.state not in advanceable_prior_phase_states:
            raise deploy_exceptions.RolloutCannotAdvanceError(rollout_name=rollout_ref.RelativeName(), failed_activity_msg='Prior phase {} is in {} state which is not advanceable.'.format(prior_phase.id, prior_phase.state))
        prompt = 'Are you sure you want to advance rollout {} to phase {}?'.format(rollout_ref.RelativeName(), ro.phases[pending_phase_index].id)
        console_io.PromptContinue(prompt, cancel_on_no=True)
        return ro.phases[pending_phase_index].id