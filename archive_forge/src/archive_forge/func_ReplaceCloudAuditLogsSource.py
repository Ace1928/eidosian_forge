from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
def ReplaceCloudAuditLogsSource(self, request, global_params=None):
    """Rpc to replace a cloudauditlogssource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudauditlogssourcesReplaceCloudAuditLogsSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudAuditLogsSource) The response message.
      """
    config = self.GetMethodConfig('ReplaceCloudAuditLogsSource')
    return self._RunMethod(config, request, global_params=global_params)