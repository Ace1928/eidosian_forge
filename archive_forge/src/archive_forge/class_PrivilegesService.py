from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class PrivilegesService(base_api.BaseApiService):
    """Service class for the privileges resource."""
    _NAME = u'privileges'

    def __init__(self, client):
        super(AdminDirectoryV1.PrivilegesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Retrieves a paginated list of all privileges for a customer.

      Args:
        request: (DirectoryPrivilegesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Privileges) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.privileges.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/roles/ALL/privileges', request_field='', request_type_name=u'DirectoryPrivilegesListRequest', response_type_name=u'Privileges', supports_download=False)