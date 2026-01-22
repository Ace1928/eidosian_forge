from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class DomainsService(base_api.BaseApiService):
    """Service class for the domains resource."""
    _NAME = u'domains'

    def __init__(self, client):
        super(AdminDirectoryV1.DomainsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a domain of the customer.

      Args:
        request: (DirectoryDomainsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryDomainsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.domains.delete', ordered_params=[u'customer', u'domainName'], path_params=[u'customer', u'domainName'], query_params=[], relative_path=u'customer/{customer}/domains/{domainName}', request_field='', request_type_name=u'DirectoryDomainsDeleteRequest', response_type_name=u'DirectoryDomainsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a domain of the customer.

      Args:
        request: (DirectoryDomainsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Domains) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.domains.get', ordered_params=[u'customer', u'domainName'], path_params=[u'customer', u'domainName'], query_params=[], relative_path=u'customer/{customer}/domains/{domainName}', request_field='', request_type_name=u'DirectoryDomainsGetRequest', response_type_name=u'Domains', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a domain of the customer.

      Args:
        request: (DirectoryDomainsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Domains) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.domains.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/domains', request_field=u'domains', request_type_name=u'DirectoryDomainsInsertRequest', response_type_name=u'Domains', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the domains of the customer.

      Args:
        request: (DirectoryDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Domains2) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.domains.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/domains', request_field='', request_type_name=u'DirectoryDomainsListRequest', response_type_name=u'Domains2', supports_download=False)