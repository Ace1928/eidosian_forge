from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class GroupsAliasesService(base_api.BaseApiService):
    """Service class for the groups_aliases resource."""
    _NAME = u'groups_aliases'

    def __init__(self, client):
        super(AdminDirectoryV1.GroupsAliasesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Remove a alias for the group.

      Args:
        request: (DirectoryGroupsAliasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryGroupsAliasesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.groups.aliases.delete', ordered_params=[u'groupKey', u'alias'], path_params=[u'alias', u'groupKey'], query_params=[], relative_path=u'groups/{groupKey}/aliases/{alias}', request_field='', request_type_name=u'DirectoryGroupsAliasesDeleteRequest', response_type_name=u'DirectoryGroupsAliasesDeleteResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Add a alias for the group.

      Args:
        request: (DirectoryGroupsAliasesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Alias) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.groups.aliases.insert', ordered_params=[u'groupKey'], path_params=[u'groupKey'], query_params=[], relative_path=u'groups/{groupKey}/aliases', request_field=u'alias', request_type_name=u'DirectoryGroupsAliasesInsertRequest', response_type_name=u'Alias', supports_download=False)

    def List(self, request, global_params=None):
        """List all aliases for a group.

      Args:
        request: (DirectoryGroupsAliasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Aliases) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.groups.aliases.list', ordered_params=[u'groupKey'], path_params=[u'groupKey'], query_params=[], relative_path=u'groups/{groupKey}/aliases', request_field='', request_type_name=u'DirectoryGroupsAliasesListRequest', response_type_name=u'Aliases', supports_download=False)