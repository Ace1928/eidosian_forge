from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.websecurityscanner.v1beta import websecurityscanner_v1beta_messages as messages
class ProjectsScanConfigsScanRunsCrawledUrlsService(base_api.BaseApiService):
    """Service class for the projects_scanConfigs_scanRuns_crawledUrls resource."""
    _NAME = 'projects_scanConfigs_scanRuns_crawledUrls'

    def __init__(self, client):
        super(WebsecurityscannerV1beta.ProjectsScanConfigsScanRunsCrawledUrlsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List CrawledUrls under a given ScanRun.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsCrawledUrlsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCrawledUrlsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns/{scanRunsId}/crawledUrls', http_method='GET', method_id='websecurityscanner.projects.scanConfigs.scanRuns.crawledUrls.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/crawledUrls', request_field='', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsCrawledUrlsListRequest', response_type_name='ListCrawledUrlsResponse', supports_download=False)