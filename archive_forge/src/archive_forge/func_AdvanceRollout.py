from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.core import log
def AdvanceRollout(self, name, phase, override_deploy_policies=None):
    """Calls the AdvanceRollout API to advance a rollout to the next phase.

    Args:
      name: Name of the Rollout. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}/rollouts/{rollout}.
      phase: The phase id on the rollout resource.
      override_deploy_policies: List of Deploy Policies to override.

    Returns:
      AdvanceRolloutResponse message.
    """
    if override_deploy_policies is None:
        override_deploy_policies = []
    request = self.messages.ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsAdvanceRequest(name=name, advanceRolloutRequest=self.messages.AdvanceRolloutRequest(phaseId=phase, overrideDeployPolicy=override_deploy_policies))
    return self._service.Advance(request)