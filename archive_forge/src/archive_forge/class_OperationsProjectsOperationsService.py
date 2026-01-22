from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class OperationsProjectsOperationsService(base_api.BaseApiService):
    """Service class for the operations_projects_operations resource."""
    _NAME = 'operations_projects_operations'

    def __init__(self, client):
        super(BigtableadminV2.OperationsProjectsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (BigtableadminOperationsProjectsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/operations/projects/{projectsId}/operations', http_method='GET', method_id='bigtableadmin.operations.projects.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+name}/operations', request_field='', request_type_name='BigtableadminOperationsProjectsOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)