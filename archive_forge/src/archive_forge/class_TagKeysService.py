from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
class TagKeysService(base_api.BaseApiService):
    """Service class for the tagKeys resource."""
    _NAME = 'tagKeys'

    def __init__(self, client):
        super(CloudresourcemanagerV3.TagKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TagKey. If another request with the same parameters is sent while the original request is in process, the second request will receive an error. A maximum of 1000 TagKeys can exist under a parent at any given time.

      Args:
        request: (CloudresourcemanagerTagKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudresourcemanager.tagKeys.create', ordered_params=[], path_params=[], query_params=['validateOnly'], relative_path='v3/tagKeys', request_field='tagKey', request_type_name='CloudresourcemanagerTagKeysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TagKey. The TagKey cannot be deleted if it has any child TagValues.

      Args:
        request: (CloudresourcemanagerTagKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}', http_method='DELETE', method_id='cloudresourcemanager.tagKeys.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagKeysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a TagKey. This method will return `PERMISSION_DENIED` if the key does not exist or the user does not have permission to view it.

      Args:
        request: (CloudresourcemanagerTagKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TagKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}', http_method='GET', method_id='cloudresourcemanager.tagKeys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='CloudresourcemanagerTagKeysGetRequest', response_type_name='TagKey', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a TagKey. The returned policy may be empty if no such policy or resource exists. The `resource` field should be the TagKey's resource name. For example, "tagKeys/1234". The caller must have `cloudresourcemanager.googleapis.com/tagKeys.getIamPolicy` permission on the specified TagKey.

      Args:
        request: (CloudresourcemanagerTagKeysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}:getIamPolicy', http_method='POST', method_id='cloudresourcemanager.tagKeys.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='CloudresourcemanagerTagKeysGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetNamespaced(self, request, global_params=None):
        """Retrieves a TagKey by its namespaced name. This method will return `PERMISSION_DENIED` if the key does not exist or the user does not have permission to view it.

      Args:
        request: (CloudresourcemanagerTagKeysGetNamespacedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TagKey) The response message.
      """
        config = self.GetMethodConfig('GetNamespaced')
        return self._RunMethod(config, request, global_params=global_params)
    GetNamespaced.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.tagKeys.getNamespaced', ordered_params=[], path_params=[], query_params=['name'], relative_path='v3/tagKeys/namespaced', request_field='', request_type_name='CloudresourcemanagerTagKeysGetNamespacedRequest', response_type_name='TagKey', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all TagKeys for a parent resource.

      Args:
        request: (CloudresourcemanagerTagKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTagKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudresourcemanager.tagKeys.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v3/tagKeys', request_field='', request_type_name='CloudresourcemanagerTagKeysListRequest', response_type_name='ListTagKeysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the attributes of the TagKey resource.

      Args:
        request: (CloudresourcemanagerTagKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}', http_method='PATCH', method_id='cloudresourcemanager.tagKeys.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v3/{+name}', request_field='tagKey', request_type_name='CloudresourcemanagerTagKeysPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a TagKey, replacing any existing policy. The `resource` field should be the TagKey's resource name. For example, "tagKeys/1234". The caller must have `resourcemanager.tagKeys.setIamPolicy` permission on the identified tagValue.

      Args:
        request: (CloudresourcemanagerTagKeysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}:setIamPolicy', http_method='POST', method_id='cloudresourcemanager.tagKeys.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudresourcemanagerTagKeysSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified TagKey. The `resource` field should be the TagKey's resource name. For example, "tagKeys/1234". There are no permissions required for making this API call.

      Args:
        request: (CloudresourcemanagerTagKeysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/tagKeys/{tagKeysId}:testIamPermissions', http_method='POST', method_id='cloudresourcemanager.tagKeys.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v3/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudresourcemanagerTagKeysTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)