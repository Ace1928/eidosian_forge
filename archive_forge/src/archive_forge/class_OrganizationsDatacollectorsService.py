from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDatacollectorsService(base_api.BaseApiService):
    """Service class for the organizations_datacollectors resource."""
    _NAME = 'organizations_datacollectors'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDatacollectorsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new data collector.

      Args:
        request: (ApigeeOrganizationsDatacollectorsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DataCollector) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/datacollectors', http_method='POST', method_id='apigee.organizations.datacollectors.create', ordered_params=['parent'], path_params=['parent'], query_params=['dataCollectorId'], relative_path='v1/{+parent}/datacollectors', request_field='googleCloudApigeeV1DataCollector', request_type_name='ApigeeOrganizationsDatacollectorsCreateRequest', response_type_name='GoogleCloudApigeeV1DataCollector', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a data collector.

      Args:
        request: (ApigeeOrganizationsDatacollectorsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/datacollectors/{datacollectorsId}', http_method='DELETE', method_id='apigee.organizations.datacollectors.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDatacollectorsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a data collector.

      Args:
        request: (ApigeeOrganizationsDatacollectorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DataCollector) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/datacollectors/{datacollectorsId}', http_method='GET', method_id='apigee.organizations.datacollectors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDatacollectorsGetRequest', response_type_name='GoogleCloudApigeeV1DataCollector', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all data collectors.

      Args:
        request: (ApigeeOrganizationsDatacollectorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDataCollectorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/datacollectors', http_method='GET', method_id='apigee.organizations.datacollectors.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/datacollectors', request_field='', request_type_name='ApigeeOrganizationsDatacollectorsListRequest', response_type_name='GoogleCloudApigeeV1ListDataCollectorsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a data collector.

      Args:
        request: (ApigeeOrganizationsDatacollectorsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DataCollector) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/datacollectors/{datacollectorsId}', http_method='PATCH', method_id='apigee.organizations.datacollectors.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DataCollector', request_type_name='ApigeeOrganizationsDatacollectorsPatchRequest', response_type_name='GoogleCloudApigeeV1DataCollector', supports_download=False)