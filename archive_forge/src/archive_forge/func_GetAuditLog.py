from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
def GetAuditLog(self, request, global_params=None):
    """Returns the requested AuditLog resource. To be deprecated.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersGetAuditLogRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1AuditLog) The response message.
      """
    config = self.GetMethodConfig('GetAuditLog')
    return self._RunMethod(config, request, global_params=global_params)