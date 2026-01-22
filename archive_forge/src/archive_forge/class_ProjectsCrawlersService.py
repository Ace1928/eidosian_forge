from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1alpha3 import datacatalog_v1alpha3_messages as messages
class ProjectsCrawlersService(base_api.BaseApiService):
    """Service class for the projects_crawlers resource."""
    _NAME = 'projects_crawlers'

    def __init__(self, client):
        super(DatacatalogV1alpha3.ProjectsCrawlersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new crawler in a project. The request fails if the crawler (`parent`, crawlerId) exists.

      Args:
        request: (DatacatalogProjectsCrawlersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Crawler) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers', http_method='POST', method_id='datacatalog.projects.crawlers.create', ordered_params=['parent'], path_params=['parent'], query_params=['crawlerId'], relative_path='v1alpha3/{+parent}/crawlers', request_field='googleCloudDatacatalogV1alpha3Crawler', request_type_name='DatacatalogProjectsCrawlersCreateRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Crawler', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a crawler in a project. The request fails if the crawler does not exist.

      Args:
        request: (DatacatalogProjectsCrawlersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}', http_method='DELETE', method_id='datacatalog.projects.crawlers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsCrawlersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a crawler in a project.

      Args:
        request: (DatacatalogProjectsCrawlersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Crawler) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}', http_method='GET', method_id='datacatalog.projects.crawlers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}', request_field='', request_type_name='DatacatalogProjectsCrawlersGetRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Crawler', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the crawlers in a project.

      Args:
        request: (DatacatalogProjectsCrawlersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3ListCrawlersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers', http_method='GET', method_id='datacatalog.projects.crawlers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha3/{+parent}/crawlers', request_field='', request_type_name='DatacatalogProjectsCrawlersListRequest', response_type_name='GoogleCloudDatacatalogV1alpha3ListCrawlersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a crawler in a project.

      Args:
        request: (DatacatalogProjectsCrawlersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3Crawler) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}', http_method='PATCH', method_id='datacatalog.projects.crawlers.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha3/{+name}', request_field='googleCloudDatacatalogV1alpha3Crawler', request_type_name='DatacatalogProjectsCrawlersPatchRequest', response_type_name='GoogleCloudDatacatalogV1alpha3Crawler', supports_download=False)

    def Run(self, request, global_params=None):
        """Runs a crawler will create and execute an ad-hoc crawler run. The request fails if the crawler is already running.

      Args:
        request: (DatacatalogProjectsCrawlersRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1alpha3CrawlerRun) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha3/projects/{projectsId}/crawlers/{crawlersId}:run', http_method='POST', method_id='datacatalog.projects.crawlers.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha3/{+name}:run', request_field='googleCloudDatacatalogV1alpha3RunCrawlerRequest', request_type_name='DatacatalogProjectsCrawlersRunRequest', response_type_name='GoogleCloudDatacatalogV1alpha3CrawlerRun', supports_download=False)