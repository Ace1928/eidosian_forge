from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securedlandingzone.v1beta import securedlandingzone_v1beta_messages as messages
class OrganizationsLocationsOperationsService(base_api.BaseApiService):
    """Service class for the organizations_locations_operations resource."""
    _NAME = 'organizations_locations_operations'

    def __init__(self, client):
        super(SecuredlandingzoneV1beta.OrganizationsLocationsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (SecuredlandingzoneOrganizationsLocationsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/operations/{operationsId}', http_method='GET', method_id='securedlandingzone.organizations.locations.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='SecuredlandingzoneOrganizationsLocationsOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)