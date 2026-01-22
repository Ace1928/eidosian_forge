from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def ReleaseSsrsLease(self, request, global_params=None):
    """Release a lease for the setup of SQL Server Reporting Services (SSRS).

      Args:
        request: (SqlInstancesReleaseSsrsLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlInstancesReleaseSsrsLeaseResponse) The response message.
      """
    config = self.GetMethodConfig('ReleaseSsrsLease')
    return self._RunMethod(config, request, global_params=global_params)