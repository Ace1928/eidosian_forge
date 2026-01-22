from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1alpha3 import datacatalog_v1alpha3_messages as messages
class ProjectsCrawlersCrawlerRunsService(base_api.BaseApiService):
    """Service class for the projects_crawlers_crawlerRuns resource."""
    _NAME = 'projects_crawlers_crawlerRuns'

    def __init__(self, client):
        super(DatacatalogV1alpha3.ProjectsCrawlersCrawlerRunsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a particular run of the crawler.

      Args:
        request: (DatacatalogProjectsCrawlersCrawlerRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3CrawlerRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}/crawlerRuns/{crawlerRunsId}', http_method='GET', method_id='datacatalog.projects.crawlers.crawlerRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsCrawlersCrawlerRunsGetRequest', response_type_name='GoogleCloudDatacatalogV1alpha3CrawlerRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists crawler runs. This includes the currently running, pending and completed crawler runs.

      Args:
        request: (DatacatalogProjectsCrawlersCrawlerRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ListCrawlerRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}/crawlerRuns', http_method='GET', method_id='datacatalog.projects.crawlers.crawlerRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha3/{+parent}/crawlerRuns', request_field='', request_type_name='DatacatalogProjectsCrawlersCrawlerRunsListRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ListCrawlerRunsResponse', supports_download=False)