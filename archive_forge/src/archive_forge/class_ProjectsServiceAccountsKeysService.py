from __future__ import absolute_import
from apitools.base.py import base_api
from samples.iam_sample.iam_v1 import iam_v1_messages as messages
class ProjectsServiceAccountsKeysService(base_api.BaseApiService):
    """Service class for the projects_serviceAccounts_keys resource."""
    _NAME = u'projects_serviceAccounts_keys'

    def __init__(self, client):
        super(IamV1.ProjectsServiceAccountsKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a ServiceAccountKey.
and returns it.

      Args:
        request: (IamProjectsServiceAccountsKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccountKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys', http_method=u'POST', method_id=u'iam.projects.serviceAccounts.keys.create', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}/keys', request_field=u'createServiceAccountKeyRequest', request_type_name=u'IamProjectsServiceAccountsKeysCreateRequest', response_type_name=u'ServiceAccountKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a ServiceAccountKey.

      Args:
        request: (IamProjectsServiceAccountsKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}', http_method=u'DELETE', method_id=u'iam.projects.serviceAccounts.keys.delete', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='', request_type_name=u'IamProjectsServiceAccountsKeysDeleteRequest', response_type_name=u'Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the ServiceAccountKey.
by key id.

      Args:
        request: (IamProjectsServiceAccountsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccountKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}', http_method=u'GET', method_id=u'iam.projects.serviceAccounts.keys.get', ordered_params=[u'name'], path_params=[u'name'], query_params=[u'publicKeyType'], relative_path=u'v1/{+name}', request_field='', request_type_name=u'IamProjectsServiceAccountsKeysGetRequest', response_type_name=u'ServiceAccountKey', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServiceAccountKeys.

      Args:
        request: (IamProjectsServiceAccountsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceAccountKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys', http_method=u'GET', method_id=u'iam.projects.serviceAccounts.keys.list', ordered_params=[u'name'], path_params=[u'name'], query_params=[u'keyTypes'], relative_path=u'v1/{+name}/keys', request_field='', request_type_name=u'IamProjectsServiceAccountsKeysListRequest', response_type_name=u'ListServiceAccountKeysResponse', supports_download=False)