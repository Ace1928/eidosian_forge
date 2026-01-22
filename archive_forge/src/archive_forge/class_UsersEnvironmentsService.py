from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudshell.v1 import cloudshell_v1_messages as messages
class UsersEnvironmentsService(base_api.BaseApiService):
    """Service class for the users_environments resource."""
    _NAME = 'users_environments'

    def __init__(self, client):
        super(CloudshellV1.UsersEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def AddPublicKey(self, request, global_params=None):
        """Adds a public SSH key to an environment, allowing clients with the corresponding private key to connect to that environment via SSH. If a key with the same content already exists, this will error with ALREADY_EXISTS.

      Args:
        request: (CloudshellUsersEnvironmentsAddPublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddPublicKey')
        return self._RunMethod(config, request, global_params=global_params)
    AddPublicKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/users/{usersId}/environments/{environmentsId}:addPublicKey', http_method='POST', method_id='cloudshell.users.environments.addPublicKey', ordered_params=['environment'], path_params=['environment'], query_params=[], relative_path='v1/{+environment}:addPublicKey', request_field='addPublicKeyRequest', request_type_name='CloudshellUsersEnvironmentsAddPublicKeyRequest', response_type_name='Operation', supports_download=False)

    def Authorize(self, request, global_params=None):
        """Sends OAuth credentials to a running environment on behalf of a user. When this completes, the environment will be authorized to run various Google Cloud command line tools without requiring the user to manually authenticate.

      Args:
        request: (CloudshellUsersEnvironmentsAuthorizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Authorize')
        return self._RunMethod(config, request, global_params=global_params)
    Authorize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/users/{usersId}/environments/{environmentsId}:authorize', http_method='POST', method_id='cloudshell.users.environments.authorize', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:authorize', request_field='authorizeEnvironmentRequest', request_type_name='CloudshellUsersEnvironmentsAuthorizeRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an environment. Returns NOT_FOUND if the environment does not exist.

      Args:
        request: (CloudshellUsersEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Environment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/users/{usersId}/environments/{environmentsId}', http_method='GET', method_id='cloudshell.users.environments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudshellUsersEnvironmentsGetRequest', response_type_name='Environment', supports_download=False)

    def RemovePublicKey(self, request, global_params=None):
        """Removes a public SSH key from an environment. Clients will no longer be able to connect to the environment using the corresponding private key. If a key with the same content is not present, this will error with NOT_FOUND.

      Args:
        request: (CloudshellUsersEnvironmentsRemovePublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemovePublicKey')
        return self._RunMethod(config, request, global_params=global_params)
    RemovePublicKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/users/{usersId}/environments/{environmentsId}:removePublicKey', http_method='POST', method_id='cloudshell.users.environments.removePublicKey', ordered_params=['environment'], path_params=['environment'], query_params=[], relative_path='v1/{+environment}:removePublicKey', request_field='removePublicKeyRequest', request_type_name='CloudshellUsersEnvironmentsRemovePublicKeyRequest', response_type_name='Operation', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts an existing environment, allowing clients to connect to it. The returned operation will contain an instance of StartEnvironmentMetadata in its metadata field. Users can wait for the environment to start by polling this operation via GetOperation. Once the environment has finished starting and is ready to accept connections, the operation will contain a StartEnvironmentResponse in its response field.

      Args:
        request: (CloudshellUsersEnvironmentsStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/users/{usersId}/environments/{environmentsId}:start', http_method='POST', method_id='cloudshell.users.environments.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:start', request_field='startEnvironmentRequest', request_type_name='CloudshellUsersEnvironmentsStartRequest', response_type_name='Operation', supports_download=False)