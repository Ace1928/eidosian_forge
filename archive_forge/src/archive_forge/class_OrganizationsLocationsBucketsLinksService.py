from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class OrganizationsLocationsBucketsLinksService(base_api.BaseApiService):
    """Service class for the organizations_locations_buckets_links resource."""
    _NAME = 'organizations_locations_buckets_links'

    def __init__(self, client):
        super(LoggingV2.OrganizationsLocationsBucketsLinksService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Asynchronously creates a linked dataset in BigQuery which makes it possible to use BigQuery to read the logs stored in the log bucket. A log bucket may currently only contain one link.

      Args:
        request: (LoggingOrganizationsLocationsBucketsLinksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/buckets/{bucketsId}/links', http_method='POST', method_id='logging.organizations.locations.buckets.links.create', ordered_params=['parent'], path_params=['parent'], query_params=['linkId'], relative_path='v2/{+parent}/links', request_field='link', request_type_name='LoggingOrganizationsLocationsBucketsLinksCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a link. This will also delete the corresponding BigQuery linked dataset.

      Args:
        request: (LoggingOrganizationsLocationsBucketsLinksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/buckets/{bucketsId}/links/{linksId}', http_method='DELETE', method_id='logging.organizations.locations.buckets.links.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingOrganizationsLocationsBucketsLinksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a link.

      Args:
        request: (LoggingOrganizationsLocationsBucketsLinksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Link) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/buckets/{bucketsId}/links/{linksId}', http_method='GET', method_id='logging.organizations.locations.buckets.links.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingOrganizationsLocationsBucketsLinksGetRequest', response_type_name='Link', supports_download=False)

    def List(self, request, global_params=None):
        """Lists links.

      Args:
        request: (LoggingOrganizationsLocationsBucketsLinksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLinksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/buckets/{bucketsId}/links', http_method='GET', method_id='logging.organizations.locations.buckets.links.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/links', request_field='', request_type_name='LoggingOrganizationsLocationsBucketsLinksListRequest', response_type_name='ListLinksResponse', supports_download=False)