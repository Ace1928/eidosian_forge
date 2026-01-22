from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privilegedaccessmanager.v1beta import privilegedaccessmanager_v1beta_messages as messages
def CheckOnboardingStatus(self, request, global_params=None):
    """CheckOnboardingStatus reports the onboarding status for a project/folder/organization. Any findings reported by this API need to be fixed before PAM can be used on the resource.

      Args:
        request: (PrivilegedaccessmanagerProjectsLocationsCheckOnboardingStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckOnboardingStatusResponse) The response message.
      """
    config = self.GetMethodConfig('CheckOnboardingStatus')
    return self._RunMethod(config, request, global_params=global_params)