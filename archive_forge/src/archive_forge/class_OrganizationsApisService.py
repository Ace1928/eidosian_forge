from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApisService(base_api.BaseApiService):
    """Service class for the organizations_apis resource."""
    _NAME = 'organizations_apis'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApisService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an API proxy. The API proxy created will not be accessible at runtime until it is deployed to an environment. Create a new API proxy by setting the `name` query parameter to the name of the API proxy. Import an API proxy configuration bundle stored in zip format on your local machine to your organization by doing the following: * Set the `name` query parameter to the name of the API proxy. * Set the `action` query parameter to `import`. * Set the `Content-Type` header to `multipart/form-data`. * Pass as a file the name of API proxy configuration bundle stored in zip format on your local machine using the `file` form field. **Note**: To validate the API proxy configuration bundle only without importing it, set the `action` query parameter to `validate`. When importing an API proxy configuration bundle, if the API proxy does not exist, it will be created. If the API proxy exists, then a new revision is created. Invalid API proxy configurations are rejected, and a list of validation errors is returned to the client.

      Args:
        request: (ApigeeOrganizationsApisCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxyRevision) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis', http_method='POST', method_id='apigee.organizations.apis.create', ordered_params=['parent'], path_params=['parent'], query_params=['action', 'name', 'validate'], relative_path='v1/{+parent}/apis', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsApisCreateRequest', response_type_name='GoogleCloudApigeeV1ApiProxyRevision', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an API proxy and all associated endpoints, policies, resources, and revisions. The API proxy must be undeployed before you can delete it.

      Args:
        request: (ApigeeOrganizationsApisDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxy) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}', http_method='DELETE', method_id='apigee.organizations.apis.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApisDeleteRequest', response_type_name='GoogleCloudApigeeV1ApiProxy', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an API proxy including a list of existing revisions.

      Args:
        request: (ApigeeOrganizationsApisGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}', http_method='GET', method_id='apigee.organizations.apis.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApisGetRequest', response_type_name='GoogleCloudApigeeV1ApiProxy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the names of all API proxies in an organization. The names returned correspond to the names defined in the configuration files for each API proxy.

      Args:
        request: (ApigeeOrganizationsApisListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListApiProxiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis', http_method='GET', method_id='apigee.organizations.apis.list', ordered_params=['parent'], path_params=['parent'], query_params=['includeMetaData', 'includeRevisions'], relative_path='v1/{+parent}/apis', request_field='', request_type_name='ApigeeOrganizationsApisListRequest', response_type_name='GoogleCloudApigeeV1ListApiProxiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing API proxy.

      Args:
        request: (ApigeeOrganizationsApisPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}', http_method='PATCH', method_id='apigee.organizations.apis.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1ApiProxy', request_type_name='ApigeeOrganizationsApisPatchRequest', response_type_name='GoogleCloudApigeeV1ApiProxy', supports_download=False)