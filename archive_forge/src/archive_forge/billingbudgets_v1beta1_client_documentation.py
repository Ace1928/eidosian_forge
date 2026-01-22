from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.billingbudgets.v1beta1 import billingbudgets_v1beta1_messages as messages
Updates a budget and returns the updated budget. WARNING: There are some fields exposed on the Google Cloud Console that aren't available on this API. Budget fields that are not exposed in this API will not be changed by this method.

      Args:
        request: (BillingbudgetsBillingAccountsBudgetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBillingBudgetsV1beta1Budget) The response message.
      