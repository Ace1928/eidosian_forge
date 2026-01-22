from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesDeploymentsService(base_api.BaseApiService):
    """Service class for the nodes_deployments resource."""
    _NAME = 'nodes_deployments'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a deployment.

      Args:
        request: (SasportalNodesDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}', http_method='DELETE', method_id='sasportal.nodes.deployments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesDeploymentsDeleteRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a requested deployment.

      Args:
        request: (SasportalNodesDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}', http_method='GET', method_id='sasportal.nodes.deployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesDeploymentsGetRequest', response_type_name='SasPortalDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists deployments.

      Args:
        request: (SasportalNodesDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments', http_method='GET', method_id='sasportal.nodes.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/deployments', request_field='', request_type_name='SasportalNodesDeploymentsListRequest', response_type_name='SasPortalListDeploymentsResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a deployment under another node or customer.

      Args:
        request: (SasportalNodesDeploymentsMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}:move', http_method='POST', method_id='sasportal.nodes.deployments.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:move', request_field='sasPortalMoveDeploymentRequest', request_type_name='SasportalNodesDeploymentsMoveRequest', response_type_name='SasPortalOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing deployment.

      Args:
        request: (SasportalNodesDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDeployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}', http_method='PATCH', method_id='sasportal.nodes.deployments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='sasPortalDeployment', request_type_name='SasportalNodesDeploymentsPatchRequest', response_type_name='SasPortalDeployment', supports_download=False)