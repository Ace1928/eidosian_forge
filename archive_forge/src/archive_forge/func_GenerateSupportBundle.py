from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
def GenerateSupportBundle(self, request, global_params=None):
    """Consumer API (private) to generate support bundles of VMware stack.

      Args:
        request: (SddcProjectsLocationsClusterGroupsGenerateSupportBundleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('GenerateSupportBundle')
    return self._RunMethod(config, request, global_params=global_params)