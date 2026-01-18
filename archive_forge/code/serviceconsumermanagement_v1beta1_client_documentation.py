from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceconsumermanagement.v1beta1 import serviceconsumermanagement_v1beta1_messages as messages
Retrieves a summary of all quota information about this consumer that is.
visible to the service producer, for each quota metric defined by the
service. Each metric includes information about all of its defined limits.
Each limit includes the limit configuration (quota unit, preciseness,
default value), the current effective limit value, and all of the overrides
applied to the limit.

      Args:
        request: (ServiceconsumermanagementServicesConsumerQuotaMetricsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V1Beta1ListConsumerQuotaMetricsResponse) The response message.
      