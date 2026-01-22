from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApisRevisionsService(base_api.BaseApiService):
    """Service class for the organizations_apis_revisions resource."""
    _NAME = 'organizations_apis_revisions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApisRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes an API proxy revision and all policies, resources, endpoints, and revisions associated with it. The API proxy revision must be undeployed before you can delete it.

      Args:
        request: (ApigeeOrganizationsApisRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxyRevision) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}/revisions/{revisionsId}', http_method='DELETE', method_id='apigee.organizations.apis.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApisRevisionsDeleteRequest', response_type_name='GoogleCloudApigeeV1ApiProxyRevision', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an API proxy revision. To download the API proxy configuration bundle for the specified revision as a zip file, set the `format` query parameter to `bundle`. If you are using curl, specify `-o filename.zip` to save the output to a file; otherwise, it displays to `stdout`. Then, develop the API proxy configuration locally and upload the updated API proxy configuration revision, as described in [updateApiProxyRevision](updateApiProxyRevision).

      Args:
        request: (ApigeeOrganizationsApisRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}/revisions/{revisionsId}', http_method='GET', method_id='apigee.organizations.apis.revisions.get', ordered_params=['name'], path_params=['name'], query_params=['format'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApisRevisionsGetRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def UpdateApiProxyRevision(self, request, global_params=None):
        """Updates an existing API proxy revision by uploading the API proxy configuration bundle as a zip file from your local machine. You can update only API proxy revisions that have never been deployed. After deployment, an API proxy revision becomes immutable, even if it is undeployed. Set the `Content-Type` header to either `multipart/form-data` or `application/octet-stream`.

      Args:
        request: (ApigeeOrganizationsApisRevisionsUpdateApiProxyRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProxyRevision) The response message.
      """
        config = self.GetMethodConfig('UpdateApiProxyRevision')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateApiProxyRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apis/{apisId}/revisions/{revisionsId}', http_method='POST', method_id='apigee.organizations.apis.revisions.updateApiProxyRevision', ordered_params=['name'], path_params=['name'], query_params=['validate'], relative_path='v1/{+name}', request_field='googleApiHttpBody', request_type_name='ApigeeOrganizationsApisRevisionsUpdateApiProxyRevisionRequest', response_type_name='GoogleCloudApigeeV1ApiProxyRevision', supports_download=False)