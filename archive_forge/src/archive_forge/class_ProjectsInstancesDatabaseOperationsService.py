from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabaseOperationsService(base_api.BaseApiService):
    """Service class for the projects_instances_databaseOperations resource."""
    _NAME = 'projects_instances_databaseOperations'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabaseOperationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists database longrunning-operations. A database operation has a name of the form `projects//instances//databases//operations/`. The long-running operation metadata field type `metadata.type_url` describes the type of the metadata. Operations returned include those that have completed/failed/canceled within the last 7 days, and pending operations.

      Args:
        request: (SpannerProjectsInstancesDatabaseOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatabaseOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databaseOperations', http_method='GET', method_id='spanner.projects.instances.databaseOperations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/databaseOperations', request_field='', request_type_name='SpannerProjectsInstancesDatabaseOperationsListRequest', response_type_name='ListDatabaseOperationsResponse', supports_download=False)