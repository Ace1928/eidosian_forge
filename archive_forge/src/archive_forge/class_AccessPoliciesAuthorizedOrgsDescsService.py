from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
class AccessPoliciesAuthorizedOrgsDescsService(base_api.BaseApiService):
    """Service class for the accessPolicies_authorizedOrgsDescs resource."""
    _NAME = 'accessPolicies_authorizedOrgsDescs'

    def __init__(self, client):
        super(AccesscontextmanagerV1alpha.AccessPoliciesAuthorizedOrgsDescsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an authorized orgs desc. The long-running operation from this RPC has a successful status after the authorized orgs desc propagates to long-lasting storage. If a authorized orgs desc contains errors, an error response is returned for the first error encountered. The name of this `AuthorizedOrgsDesc` will be assigned during creation.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/authorizedOrgsDescs', http_method='POST', method_id='accesscontextmanager.accessPolicies.authorizedOrgsDescs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/authorizedOrgsDescs', request_field='authorizedOrgsDesc', request_type_name='AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an authorized orgs desc based on the resource name. The long-running operation from this RPC has a successful status after the authorized orgs desc is removed from long-lasting storage.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/authorizedOrgsDescs/{authorizedOrgsDescsId}', http_method='DELETE', method_id='accesscontextmanager.accessPolicies.authorizedOrgsDescs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an authorized orgs desc based on the resource name.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizedOrgsDesc) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/authorizedOrgsDescs/{authorizedOrgsDescsId}', http_method='GET', method_id='accesscontextmanager.accessPolicies.authorizedOrgsDescs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsGetRequest', response_type_name='AuthorizedOrgsDesc', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all authorized orgs descs for an access policy.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedOrgsDescsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/authorizedOrgsDescs', http_method='GET', method_id='accesscontextmanager.accessPolicies.authorizedOrgsDescs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/authorizedOrgsDescs', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsListRequest', response_type_name='ListAuthorizedOrgsDescsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an authorized orgs desc. The long-running operation from this RPC has a successful status after the authorized orgs desc propagates to long-lasting storage. If a authorized orgs desc contains errors, an error response is returned for the first error encountered. Only the organization list in `AuthorizedOrgsDesc` can be updated. The name, authorization_type, asset_type and authorization_direction cannot be updated.

      Args:
        request: (AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}/authorizedOrgsDescs/{authorizedOrgsDescsId}', http_method='PATCH', method_id='accesscontextmanager.accessPolicies.authorizedOrgsDescs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='authorizedOrgsDesc', request_type_name='AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsPatchRequest', response_type_name='Operation', supports_download=False)