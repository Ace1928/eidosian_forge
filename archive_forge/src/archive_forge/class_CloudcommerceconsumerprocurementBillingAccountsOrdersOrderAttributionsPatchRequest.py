from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsPatchRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributions
  PatchRequest object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1OrderAttribution: A
      GoogleCloudCommerceConsumerProcurementV1alpha1OrderAttribution resource
      to be passed as the request body.
    name: Output only. Resource name of the attribution configuration Format:
      billingAccounts/{billing_account}/orders/{order}/orderAttributions/{orde
      r_attribution} attribution_target references the Order parameter that
      defines the total attributable amount of this resource.
    updateMask: Optional. Mask used to indicate which parts of
      OrderAttribution are to be updated.
  """
    googleCloudCommerceConsumerProcurementV1alpha1OrderAttribution = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1OrderAttribution', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)