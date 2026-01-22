from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
class TabledataService(base_api.BaseApiService):
    """Service class for the tabledata resource."""
    _NAME = u'tabledata'

    def __init__(self, client):
        super(BigqueryV2.TabledataService, self).__init__(client)
        self._upload_configs = {}

    def InsertAll(self, request, global_params=None):
        """Streams data into BigQuery one record at a time without needing to run a load job. Requires the WRITER dataset role.

      Args:
        request: (BigqueryTabledataInsertAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TableDataInsertAllResponse) The response message.
      """
        config = self.GetMethodConfig('InsertAll')
        return self._RunMethod(config, request, global_params=global_params)
    InsertAll.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.tabledata.insertAll', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}/insertAll', request_field=u'tableDataInsertAllRequest', request_type_name=u'BigqueryTabledataInsertAllRequest', response_type_name=u'TableDataInsertAllResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves table data from a specified set of rows. Requires the READER dataset role.

      Args:
        request: (BigqueryTabledataListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TableDataList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.tabledata.list', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[u'maxResults', u'pageToken', u'startIndex'], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data', request_field='', request_type_name=u'BigqueryTabledataListRequest', response_type_name=u'TableDataList', supports_download=False)