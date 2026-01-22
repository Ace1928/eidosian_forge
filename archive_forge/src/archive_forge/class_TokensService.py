from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class TokensService(base_api.BaseApiService):
    """Service class for the tokens resource."""
    _NAME = u'tokens'

    def __init__(self, client):
        super(AdminDirectoryV1.TokensService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete all access tokens issued by a user for an application.

      Args:
        request: (DirectoryTokensDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryTokensDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.tokens.delete', ordered_params=[u'userKey', u'clientId'], path_params=[u'clientId', u'userKey'], query_params=[], relative_path=u'users/{userKey}/tokens/{clientId}', request_field='', request_type_name=u'DirectoryTokensDeleteRequest', response_type_name=u'DirectoryTokensDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about an access token issued by a user.

      Args:
        request: (DirectoryTokensGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Token) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.tokens.get', ordered_params=[u'userKey', u'clientId'], path_params=[u'clientId', u'userKey'], query_params=[], relative_path=u'users/{userKey}/tokens/{clientId}', request_field='', request_type_name=u'DirectoryTokensGetRequest', response_type_name=u'Token', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the set of tokens specified user has issued to 3rd party applications.

      Args:
        request: (DirectoryTokensListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Tokens) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.tokens.list', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/tokens', request_field='', request_type_name=u'DirectoryTokensListRequest', response_type_name=u'Tokens', supports_download=False)