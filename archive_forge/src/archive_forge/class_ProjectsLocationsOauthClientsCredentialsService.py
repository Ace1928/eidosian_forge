from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsOauthClientsCredentialsService(base_api.BaseApiService):
    """Service class for the projects_locations_oauthClients_credentials resource."""
    _NAME = 'projects_locations_oauthClients_credentials'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsOauthClientsCredentialsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new OauthClientCredential.

      Args:
        request: (IamProjectsLocationsOauthClientsCredentialsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClientCredential) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}/credentials', http_method='POST', method_id='iam.projects.locations.oauthClients.credentials.create', ordered_params=['parent'], path_params=['parent'], query_params=['oauthClientCredentialId'], relative_path='v1/{+parent}/credentials', request_field='oauthClientCredential', request_type_name='IamProjectsLocationsOauthClientsCredentialsCreateRequest', response_type_name='OauthClientCredential', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a OauthClientCredential. Before deleting an oauth client credential, it should first be disabled.

      Args:
        request: (IamProjectsLocationsOauthClientsCredentialsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}/credentials/{credentialsId}', http_method='DELETE', method_id='iam.projects.locations.oauthClients.credentials.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsOauthClientsCredentialsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual OauthClientCredential.

      Args:
        request: (IamProjectsLocationsOauthClientsCredentialsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClientCredential) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}/credentials/{credentialsId}', http_method='GET', method_id='iam.projects.locations.oauthClients.credentials.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsOauthClientsCredentialsGetRequest', response_type_name='OauthClientCredential', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all OauthClientCredentialss in a OauthClient.

      Args:
        request: (IamProjectsLocationsOauthClientsCredentialsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOauthClientCredentialsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}/credentials', http_method='GET', method_id='iam.projects.locations.oauthClients.credentials.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/credentials', request_field='', request_type_name='IamProjectsLocationsOauthClientsCredentialsListRequest', response_type_name='ListOauthClientCredentialsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing OauthClientCredential.

      Args:
        request: (IamProjectsLocationsOauthClientsCredentialsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClientCredential) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}/credentials/{credentialsId}', http_method='PATCH', method_id='iam.projects.locations.oauthClients.credentials.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='oauthClientCredential', request_type_name='IamProjectsLocationsOauthClientsCredentialsPatchRequest', response_type_name='OauthClientCredential', supports_download=False)