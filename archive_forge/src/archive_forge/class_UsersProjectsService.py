from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
class UsersProjectsService(base_api.BaseApiService):
    """Service class for the users_projects resource."""
    _NAME = 'users_projects'

    def __init__(self, client):
        super(OsloginV1beta.UsersProjectsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a POSIX account.

      Args:
        request: (OsloginUsersProjectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/projects/{projectsId}', http_method='DELETE', method_id='oslogin.users.projects.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsloginUsersProjectsDeleteRequest', response_type_name='Empty', supports_download=False)