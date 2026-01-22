from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class FoldersLocationsOperationIdsService(base_api.BaseApiService):
    """Service class for the folders_locations_operationIds resource."""
    _NAME = 'folders_locations_operationIds'

    def __init__(self, client):
        super(AuditmanagerV1alpha.FoldersLocationsOperationIdsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details about generate audit report operation.

      Args:
        request: (AuditmanagerFoldersLocationsOperationIdsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/operationIds/{operationIdsId}', http_method='GET', method_id='auditmanager.folders.locations.operationIds.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerFoldersLocationsOperationIdsGetRequest', response_type_name='Operation', supports_download=False)