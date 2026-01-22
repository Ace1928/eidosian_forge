from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesNodesNodesService(base_api.BaseApiService):
    """Service class for the nodes_nodes_nodes resource."""
    _NAME = 'nodes_nodes_nodes'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesNodesNodesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new node.

      Args:
        request: (SasportalNodesNodesNodesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}/nodes', http_method='POST', method_id='sasportal.nodes.nodes.nodes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/nodes', request_field='sasPortalNode', request_type_name='SasportalNodesNodesNodesCreateRequest', response_type_name='SasPortalNode', supports_download=False)

    def List(self, request, global_params=None):
        """Lists nodes.

      Args:
        request: (SasportalNodesNodesNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}/nodes', http_method='GET', method_id='sasportal.nodes.nodes.nodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/nodes', request_field='', request_type_name='SasportalNodesNodesNodesListRequest', response_type_name='SasPortalListNodesResponse', supports_download=False)