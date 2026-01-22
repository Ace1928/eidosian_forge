from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class InfoTypesService(base_api.BaseApiService):
    """Service class for the infoTypes resource."""
    _NAME = 'infoTypes'

    def __init__(self, client):
        super(DlpV2.InfoTypesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns a list of the sensitive information types that DLP API supports. See https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference to learn more.

      Args:
        request: (DlpInfoTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListInfoTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dlp.infoTypes.list', ordered_params=[], path_params=[], query_params=['filter', 'languageCode', 'locationId', 'parent'], relative_path='v2/infoTypes', request_field='', request_type_name='DlpInfoTypesListRequest', response_type_name='GooglePrivacyDlpV2ListInfoTypesResponse', supports_download=False)