from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.websecurityscanner.v1beta import websecurityscanner_v1beta_messages as messages
class ProjectsScanConfigsScanRunsService(base_api.BaseApiService):
    """Service class for the projects_scanConfigs_scanRuns resource."""
    _NAME = 'projects_scanConfigs_scanRuns'

    def __init__(self, client):
        super(WebsecurityscannerV1beta.ProjectsScanConfigsScanRunsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a ScanRun.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScanRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns/{scanRunsId}', http_method='GET', method_id='websecurityscanner.projects.scanConfigs.scanRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsGetRequest', response_type_name='ScanRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ScanRuns under a given ScanConfig, in descending order of ScanRun stop time.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScanRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns', http_method='GET', method_id='websecurityscanner.projects.scanConfigs.scanRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/scanRuns', request_field='', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsListRequest', response_type_name='ListScanRunsResponse', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a ScanRun. The stopped ScanRun is returned.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScanRun) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns/{scanRunsId}:stop', http_method='POST', method_id='websecurityscanner.projects.scanConfigs.scanRuns.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:stop', request_field='stopScanRunRequest', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsStopRequest', response_type_name='ScanRun', supports_download=False)