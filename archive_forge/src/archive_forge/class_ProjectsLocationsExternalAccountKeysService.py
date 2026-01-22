from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.publicca.v1 import publicca_v1_messages as messages
class ProjectsLocationsExternalAccountKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_externalAccountKeys resource."""
    _NAME = 'projects_locations_externalAccountKeys'

    def __init__(self, client):
        super(PubliccaV1.ProjectsLocationsExternalAccountKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ExternalAccountKey bound to the project.

      Args:
        request: (PubliccaProjectsLocationsExternalAccountKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalAccountKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/externalAccountKeys', http_method='POST', method_id='publicca.projects.locations.externalAccountKeys.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/externalAccountKeys', request_field='externalAccountKey', request_type_name='PubliccaProjectsLocationsExternalAccountKeysCreateRequest', response_type_name='ExternalAccountKey', supports_download=False)