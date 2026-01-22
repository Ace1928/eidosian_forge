from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class ProjectsEntitlementsService(base_api.BaseApiService):
    """Service class for the projects_entitlements resource."""
    _NAME = 'projects_entitlements'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.ProjectsEntitlementsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the requested Entitlement resource.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsEntitlementsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1Entitlement) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/entitlements/{entitlementsId}', http_method='GET', method_id='cloudcommerceconsumerprocurement.projects.entitlements.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementProjectsEntitlementsGetRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1Entitlement', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Entitlement resources that the user has access to, within the scope of the parent resource. This includes all Entitlements that are either parented by a billing account associated with the parent (project) and or the project is a consumer of an Order.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsEntitlementsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListEntitlementsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/entitlements', http_method='GET', method_id='cloudcommerceconsumerprocurement.projects.entitlements.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/entitlements', request_field='', request_type_name='CloudcommerceconsumerprocurementProjectsEntitlementsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListEntitlementsResponse', supports_download=False)