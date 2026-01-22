from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firebasedataconnect.v1alpha import firebasedataconnect_v1alpha_messages as messages
class ProjectsLocationsServicesConnectorsRevisionsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_connectors_revisions resource."""
    _NAME = 'projects_locations_services_connectors_revisions'

    def __init__(self, client):
        super(FirebasedataconnectV1alpha.ProjectsLocationsServicesConnectorsRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a single ConnectorRevision.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}/revisions/{revisionsId}', http_method='DELETE', method_id='firebasedataconnect.projects.locations.services.connectors.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ConnectorRevision.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectorRevision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}/revisions/{revisionsId}', http_method='GET', method_id='firebasedataconnect.projects.locations.services.connectors.revisions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsGetRequest', response_type_name='ConnectorRevision', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ConnectorRevisions in a given project and location.

      Args:
        request: (FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectorRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/connectors/{connectorsId}/revisions', http_method='GET', method_id='firebasedataconnect.projects.locations.services.connectors.revisions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/revisions', request_field='', request_type_name='FirebasedataconnectProjectsLocationsServicesConnectorsRevisionsListRequest', response_type_name='ListConnectorRevisionsResponse', supports_download=False)