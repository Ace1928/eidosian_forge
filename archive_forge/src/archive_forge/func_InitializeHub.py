from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha2 import gkehub_v1alpha2_messages as messages
def InitializeHub(self, request, global_params=None):
    """Initializes the Hub in this project, which includes creating the default Hub Service Account and the Hub Workload Identity Pool. Initialization is optional, and happens automatically when the first Membership is created. InitializeHub should be called when the first Membership cannot be registered without these resources. A common example is granting the Hub Service Account access to another project, which requires the account to exist first.

      Args:
        request: (GkehubProjectsLocationsGlobalMembershipsInitializeHubRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InitializeHubResponse) The response message.
      """
    config = self.GetMethodConfig('InitializeHub')
    return self._RunMethod(config, request, global_params=global_params)