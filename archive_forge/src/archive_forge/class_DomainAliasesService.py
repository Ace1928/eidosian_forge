from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class DomainAliasesService(base_api.BaseApiService):
    """Service class for the domainAliases resource."""
    _NAME = u'domainAliases'

    def __init__(self, client):
        super(AdminDirectoryV1.DomainAliasesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a Domain Alias of the customer.

      Args:
        request: (DirectoryDomainAliasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryDomainAliasesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.domainAliases.delete', ordered_params=[u'customer', u'domainAliasName'], path_params=[u'customer', u'domainAliasName'], query_params=[], relative_path=u'customer/{customer}/domainaliases/{domainAliasName}', request_field='', request_type_name=u'DirectoryDomainAliasesDeleteRequest', response_type_name=u'DirectoryDomainAliasesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a domain alias of the customer.

      Args:
        request: (DirectoryDomainAliasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DomainAlias) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.domainAliases.get', ordered_params=[u'customer', u'domainAliasName'], path_params=[u'customer', u'domainAliasName'], query_params=[], relative_path=u'customer/{customer}/domainaliases/{domainAliasName}', request_field='', request_type_name=u'DirectoryDomainAliasesGetRequest', response_type_name=u'DomainAlias', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a Domain alias of the customer.

      Args:
        request: (DirectoryDomainAliasesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DomainAlias) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.domainAliases.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/domainaliases', request_field=u'domainAlias', request_type_name=u'DirectoryDomainAliasesInsertRequest', response_type_name=u'DomainAlias', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the domain aliases of the customer.

      Args:
        request: (DirectoryDomainAliasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DomainAliases) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.domainAliases.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'parentDomainName'], relative_path=u'customer/{customer}/domainaliases', request_field='', request_type_name=u'DirectoryDomainAliasesListRequest', response_type_name=u'DomainAliases', supports_download=False)