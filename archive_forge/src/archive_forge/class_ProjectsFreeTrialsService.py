from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class ProjectsFreeTrialsService(base_api.BaseApiService):
    """Service class for the projects_freeTrials resource."""
    _NAME = 'projects_freeTrials'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.ProjectsFreeTrialsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FreeTrial.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsFreeTrialsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/freeTrials', http_method='POST', method_id='cloudcommerceconsumerprocurement.projects.freeTrials.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/freeTrials', request_field='googleCloudCommerceConsumerProcurementV1alpha1FreeTrial', request_type_name='CloudcommerceconsumerprocurementProjectsFreeTrialsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the requested FreeTrial resource.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsFreeTrialsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/freeTrials/{freeTrialsId}', http_method='GET', method_id='cloudcommerceconsumerprocurement.projects.freeTrials.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementProjectsFreeTrialsGetRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FreeTrial resources that the user has access to, within the scope of the parent resource.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsFreeTrialsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListFreeTrialsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/freeTrials', http_method='GET', method_id='cloudcommerceconsumerprocurement.projects.freeTrials.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/freeTrials', request_field='', request_type_name='CloudcommerceconsumerprocurementProjectsFreeTrialsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListFreeTrialsResponse', supports_download=False)