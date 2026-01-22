from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
def IgnoreJob(self, request, global_params=None):
    """Ignores the specified Job in a Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsIgnoreJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IgnoreJobResponse) The response message.
      """
    config = self.GetMethodConfig('IgnoreJob')
    return self._RunMethod(config, request, global_params=global_params)