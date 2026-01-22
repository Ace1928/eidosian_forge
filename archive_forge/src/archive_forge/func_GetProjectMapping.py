from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetProjectMapping(self, request, global_params=None):
    """Gets the project ID and region for an Apigee organization.

      Args:
        request: (ApigeeOrganizationsGetProjectMappingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1OrganizationProjectMapping) The response message.
      """
    config = self.GetMethodConfig('GetProjectMapping')
    return self._RunMethod(config, request, global_params=global_params)