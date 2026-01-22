from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privilegedaccessmanager.v1beta import privilegedaccessmanager_v1beta_messages as messages
class FoldersLocationsEntitlementsGrantsService(base_api.BaseApiService):
    """Service class for the folders_locations_entitlements_grants resource."""
    _NAME = 'folders_locations_entitlements_grants'

    def __init__(self, client):
        super(PrivilegedaccessmanagerV1beta.FoldersLocationsEntitlementsGrantsService, self).__init__(client)
        self._upload_configs = {}

    def Approve(self, request, global_params=None):
        """ApproveGrant is used to provide approval for a Grant. This method can only be called while the Grant is in `APPROVAL_AWAITED` state. An approval once granted on a Grant can not be taken back.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Grant) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants/{grantsId}:approve', http_method='POST', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:approve', request_field='approveGrantRequest', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsApproveRequest', response_type_name='Grant', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Grant in a given project and location.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Grant) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants', http_method='POST', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1beta/{+parent}/grants', request_field='grant', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsCreateRequest', response_type_name='Grant', supports_download=False)

    def Deny(self, request, global_params=None):
        """DenyGrant is like ApproveGrant but is used for denying a Grant. This method can only be called while the Grant is in `APPROVAL_AWAITED` state. This operation cannot be undone.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsDenyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Grant) The response message.
      """
        config = self.GetMethodConfig('Deny')
        return self._RunMethod(config, request, global_params=global_params)
    Deny.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants/{grantsId}:deny', http_method='POST', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.deny', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:deny', request_field='denyGrantRequest', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsDenyRequest', response_type_name='Grant', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details of a single Grant.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Grant) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants/{grantsId}', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsGetRequest', response_type_name='Grant', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Grants for a given Entitlement.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGrantsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/grants', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsListRequest', response_type_name='ListGrantsResponse', supports_download=False)

    def Revoke(self, request, global_params=None):
        """RevokeGrant is used to immediately revoke access for a Grant. This method can be called when the Grant is in a non terminal state.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Revoke')
        return self._RunMethod(config, request, global_params=global_params)
    Revoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants/{grantsId}:revoke', http_method='POST', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.revoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:revoke', request_field='revokeGrantRequest', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsRevokeRequest', response_type_name='Operation', supports_download=False)

    def Search(self, request, global_params=None):
        """SearchGrants returns Grants that are related to the calling user in the specified way.

      Args:
        request: (PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchGrantsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/folders/{foldersId}/locations/{locationsId}/entitlements/{entitlementsId}/grants:search', http_method='GET', method_id='privilegedaccessmanager.folders.locations.entitlements.grants.search', ordered_params=['parent'], path_params=['parent'], query_params=['callerRelationship', 'filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/grants:search', request_field='', request_type_name='PrivilegedaccessmanagerFoldersLocationsEntitlementsGrantsSearchRequest', response_type_name='SearchGrantsResponse', supports_download=False)