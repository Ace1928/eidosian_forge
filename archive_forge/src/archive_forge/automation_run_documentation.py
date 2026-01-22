from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
Cancels an automation run.

    Args:
      name: Name of the AutomationRun. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/automationRuns/{automationRun}.

    Returns:
      CancelAutomationRunResponse message.
    