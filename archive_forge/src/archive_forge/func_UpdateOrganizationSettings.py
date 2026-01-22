from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
def UpdateOrganizationSettings(self, request, global_params=None):
    """Updates an organization's settings.

      Args:
        request: (SecuritycenterOrganizationsUpdateOrganizationSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrganizationSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateOrganizationSettings')
    return self._RunMethod(config, request, global_params=global_params)