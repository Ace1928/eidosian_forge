from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
class ProjectsConfigsWaitersService(base_api.BaseApiService):
    """Service class for the projects_configs_waiters resource."""
    _NAME = 'projects_configs_waiters'

    def __init__(self, client):
        super(RuntimeconfigV1beta1.ProjectsConfigsWaitersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Waiter resource. This operation returns a long-running Operation resource which can be polled for completion. However, a waiter with the given name will exist (and can be retrieved) prior to the operation completing. If the operation fails, the failed Waiter resource will still exist and must be deleted prior to subsequent creation attempts.

      Args:
        request: (RuntimeconfigProjectsConfigsWaitersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/waiters', http_method='POST', method_id='runtimeconfig.projects.configs.waiters.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1beta1/{+parent}/waiters', request_field='waiter', request_type_name='RuntimeconfigProjectsConfigsWaitersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the waiter with the specified name.

      Args:
        request: (RuntimeconfigProjectsConfigsWaitersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/waiters/{waitersId}', http_method='DELETE', method_id='runtimeconfig.projects.configs.waiters.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsWaitersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a single waiter.

      Args:
        request: (RuntimeconfigProjectsConfigsWaitersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Waiter) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/waiters/{waitersId}', http_method='GET', method_id='runtimeconfig.projects.configs.waiters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='RuntimeconfigProjectsConfigsWaitersGetRequest', response_type_name='Waiter', supports_download=False)

    def List(self, request, global_params=None):
        """List waiters within the given configuration.

      Args:
        request: (RuntimeconfigProjectsConfigsWaitersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWaitersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/waiters', http_method='GET', method_id='runtimeconfig.projects.configs.waiters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/waiters', request_field='', request_type_name='RuntimeconfigProjectsConfigsWaitersListRequest', response_type_name='ListWaitersResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (RuntimeconfigProjectsConfigsWaitersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/configs/{configsId}/waiters/{waitersId}:testIamPermissions', http_method='POST', method_id='runtimeconfig.projects.configs.waiters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='RuntimeconfigProjectsConfigsWaitersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)