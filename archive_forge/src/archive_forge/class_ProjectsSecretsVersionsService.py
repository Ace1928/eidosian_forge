from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.secretmanager.v1 import secretmanager_v1_messages as messages
class ProjectsSecretsVersionsService(base_api.BaseApiService):
    """Service class for the projects_secrets_versions resource."""
    _NAME = 'projects_secrets_versions'

    def __init__(self, client):
        super(SecretmanagerV1.ProjectsSecretsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Access(self, request, global_params=None):
        """Accesses a SecretVersion. This call returns the secret data. `projects/*/secrets/*/versions/latest` is an alias to the most recently created SecretVersion.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSecretVersionResponse) The response message.
      """
        config = self.GetMethodConfig('Access')
        return self._RunMethod(config, request, global_params=global_params)
    Access.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions/{versionsId}:access', http_method='GET', method_id='secretmanager.projects.secrets.versions.access', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:access', request_field='', request_type_name='SecretmanagerProjectsSecretsVersionsAccessRequest', response_type_name='AccessSecretVersionResponse', supports_download=False)

    def Destroy(self, request, global_params=None):
        """Destroys a SecretVersion. Sets the state of the SecretVersion to DESTROYED and irrevocably destroys the secret data.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsDestroyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecretVersion) The response message.
      """
        config = self.GetMethodConfig('Destroy')
        return self._RunMethod(config, request, global_params=global_params)
    Destroy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions/{versionsId}:destroy', http_method='POST', method_id='secretmanager.projects.secrets.versions.destroy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:destroy', request_field='destroySecretVersionRequest', request_type_name='SecretmanagerProjectsSecretsVersionsDestroyRequest', response_type_name='SecretVersion', supports_download=False)

    def Disable(self, request, global_params=None):
        """Disables a SecretVersion. Sets the state of the SecretVersion to DISABLED.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecretVersion) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions/{versionsId}:disable', http_method='POST', method_id='secretmanager.projects.secrets.versions.disable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:disable', request_field='disableSecretVersionRequest', request_type_name='SecretmanagerProjectsSecretsVersionsDisableRequest', response_type_name='SecretVersion', supports_download=False)

    def Enable(self, request, global_params=None):
        """Enables a SecretVersion. Sets the state of the SecretVersion to ENABLED.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecretVersion) The response message.
      """
        config = self.GetMethodConfig('Enable')
        return self._RunMethod(config, request, global_params=global_params)
    Enable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions/{versionsId}:enable', http_method='POST', method_id='secretmanager.projects.secrets.versions.enable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:enable', request_field='enableSecretVersionRequest', request_type_name='SecretmanagerProjectsSecretsVersionsEnableRequest', response_type_name='SecretVersion', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets metadata for a SecretVersion. `projects/*/secrets/*/versions/latest` is an alias to the most recently created SecretVersion.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecretVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions/{versionsId}', http_method='GET', method_id='secretmanager.projects.secrets.versions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecretmanagerProjectsSecretsVersionsGetRequest', response_type_name='SecretVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SecretVersions. This call does not return secret data.

      Args:
        request: (SecretmanagerProjectsSecretsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecretVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/secrets/{secretsId}/versions', http_method='GET', method_id='secretmanager.projects.secrets.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/versions', request_field='', request_type_name='SecretmanagerProjectsSecretsVersionsListRequest', response_type_name='ListSecretVersionsResponse', supports_download=False)