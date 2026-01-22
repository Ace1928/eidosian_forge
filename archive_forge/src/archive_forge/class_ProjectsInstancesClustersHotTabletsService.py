from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesClustersHotTabletsService(base_api.BaseApiService):
    """Service class for the projects_instances_clusters_hotTablets resource."""
    _NAME = 'projects_instances_clusters_hotTablets'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesClustersHotTabletsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists hot tablets in a cluster, within the time range provided. Hot tablets are ordered based on CPU usage.

      Args:
        request: (BigtableadminProjectsInstancesClustersHotTabletsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHotTabletsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/hotTablets', http_method='GET', method_id='bigtableadmin.projects.instances.clusters.hotTablets.list', ordered_params=['parent'], path_params=['parent'], query_params=['endTime', 'pageSize', 'pageToken', 'startTime'], relative_path='v2/{+parent}/hotTablets', request_field='', request_type_name='BigtableadminProjectsInstancesClustersHotTabletsListRequest', response_type_name='ListHotTabletsResponse', supports_download=False)