from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
class ServicesConsumerQuotaMetricsLimitsService(base_api.BaseApiService):
    """Service class for the services_consumerQuotaMetrics_limits resource."""
    _NAME = 'services_consumerQuotaMetrics_limits'

    def __init__(self, client):
        super(ServiceusageV1beta1.ServicesConsumerQuotaMetricsLimitsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a summary of quota information for a specific quota limit.

      Args:
        request: (ServiceusageServicesConsumerQuotaMetricsLimitsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsumerQuotaLimit) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}/{v1beta1Id1}/services/{servicesId}/consumerQuotaMetrics/{consumerQuotaMetricsId}/limits/{limitsId}', http_method='GET', method_id='serviceusage.services.consumerQuotaMetrics.limits.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1beta1/{+name}', request_field='', request_type_name='ServiceusageServicesConsumerQuotaMetricsLimitsGetRequest', response_type_name='ConsumerQuotaLimit', supports_download=False)