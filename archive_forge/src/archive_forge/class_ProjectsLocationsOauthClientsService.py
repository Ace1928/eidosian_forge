from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsLocationsOauthClientsService(base_api.BaseApiService):
    """Service class for the projects_locations_oauthClients resource."""
    _NAME = 'projects_locations_oauthClients'

    def __init__(self, client):
        super(IamV1.ProjectsLocationsOauthClientsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new OauthClient. You cannot reuse the name of a deleted oauth client until 30 days after deletion.

      Args:
        request: (IamProjectsLocationsOauthClientsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClient) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients', http_method='POST', method_id='iam.projects.locations.oauthClients.create', ordered_params=['parent'], path_params=['parent'], query_params=['oauthClientId'], relative_path='v1/{+parent}/oauthClients', request_field='oauthClient', request_type_name='IamProjectsLocationsOauthClientsCreateRequest', response_type_name='OauthClient', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a OauthClient. You cannot use a deleted oauth client. However, deletion does not revoke access tokens that have already been issued; they continue to grant access. Deletion does revoke refresh tokens that have already been issued; They cannot be used to renew an access token. If the oauth client is undeleted, and the refresh tokens are not expired, they are valid for token exchange again. You can undelete an oauth client for 30 days. After 30 days, deletion is permanent. You cannot update deleted oauth clients. However, you can view and list them.

      Args:
        request: (IamProjectsLocationsOauthClientsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClient) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}', http_method='DELETE', method_id='iam.projects.locations.oauthClients.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsOauthClientsDeleteRequest', response_type_name='OauthClient', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual OauthClient.

      Args:
        request: (IamProjectsLocationsOauthClientsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClient) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}', http_method='GET', method_id='iam.projects.locations.oauthClients.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsLocationsOauthClientsGetRequest', response_type_name='OauthClient', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted OauthClientss in a project. If `show_deleted` is set to `true`, then deleted oauth clients are also listed.

      Args:
        request: (IamProjectsLocationsOauthClientsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOauthClientsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients', http_method='GET', method_id='iam.projects.locations.oauthClients.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/oauthClients', request_field='', request_type_name='IamProjectsLocationsOauthClientsListRequest', response_type_name='ListOauthClientsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing OauthClient.

      Args:
        request: (IamProjectsLocationsOauthClientsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClient) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}', http_method='PATCH', method_id='iam.projects.locations.oauthClients.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='oauthClient', request_type_name='IamProjectsLocationsOauthClientsPatchRequest', response_type_name='OauthClient', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a OauthClient, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsOauthClientsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OauthClient) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/oauthClients/{oauthClientsId}:undelete', http_method='POST', method_id='iam.projects.locations.oauthClients.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteOauthClientRequest', request_type_name='IamProjectsLocationsOauthClientsUndeleteRequest', response_type_name='OauthClient', supports_download=False)