from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
class ColumnService(base_api.BaseApiService):
    """Service class for the column resource."""
    _NAME = u'column'

    def __init__(self, client):
        super(FusiontablesV1.ColumnService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the column.

      Args:
        request: (FusiontablesColumnDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FusiontablesColumnDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'fusiontables.column.delete', ordered_params=[u'tableId', u'columnId'], path_params=[u'columnId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/columns/{columnId}', request_field='', request_type_name=u'FusiontablesColumnDeleteRequest', response_type_name=u'FusiontablesColumnDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific column by its id.

      Args:
        request: (FusiontablesColumnGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Column) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.column.get', ordered_params=[u'tableId', u'columnId'], path_params=[u'columnId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/columns/{columnId}', request_field='', request_type_name=u'FusiontablesColumnGetRequest', response_type_name=u'Column', supports_download=False)

    def Insert(self, request, global_params=None):
        """Adds a new column to the table.

      Args:
        request: (FusiontablesColumnInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Column) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.column.insert', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[], relative_path=u'tables/{tableId}/columns', request_field=u'column', request_type_name=u'FusiontablesColumnInsertRequest', response_type_name=u'Column', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of columns.

      Args:
        request: (FusiontablesColumnListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ColumnList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.column.list', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'maxResults', u'pageToken'], relative_path=u'tables/{tableId}/columns', request_field='', request_type_name=u'FusiontablesColumnListRequest', response_type_name=u'ColumnList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the name or type of an existing column. This method supports patch semantics.

      Args:
        request: (FusiontablesColumnPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Column) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'fusiontables.column.patch', ordered_params=[u'tableId', u'columnId'], path_params=[u'columnId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/columns/{columnId}', request_field=u'column', request_type_name=u'FusiontablesColumnPatchRequest', response_type_name=u'Column', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the name or type of an existing column.

      Args:
        request: (FusiontablesColumnUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Column) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'fusiontables.column.update', ordered_params=[u'tableId', u'columnId'], path_params=[u'columnId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/columns/{columnId}', request_field=u'column', request_type_name=u'FusiontablesColumnUpdateRequest', response_type_name=u'Column', supports_download=False)