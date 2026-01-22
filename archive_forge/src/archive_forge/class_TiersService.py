from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class TiersService(base_api.BaseApiService):
    """Service class for the tiers resource."""
    _NAME = 'tiers'

    def __init__(self, client):
        super(SqladminV1beta4.TiersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all available machine types (tiers) for Cloud SQL, for example, `db-custom-1-3840`. For related information, see [Pricing](/sql/pricing).

      Args:
        request: (SqlTiersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TiersListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.tiers.list', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/tiers', request_field='', request_type_name='SqlTiersListRequest', response_type_name='TiersListResponse', supports_download=False)