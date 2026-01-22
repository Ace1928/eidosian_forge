from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsConsentsRevokeRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsConsentsRevokeRequest
  object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1RevokeConsentRequest: A
      GoogleCloudCommerceConsumerProcurementV1alpha1RevokeConsentRequest
      resource to be passed as the request body.
    name: Required. A consent to be reovked. Examples of valid names would be:
      - billingAccounts/{billing_account}/consents/{consent_id} -
      projects/{project_id}/consents/{consent_id}
  """
    googleCloudCommerceConsumerProcurementV1alpha1RevokeConsentRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1RevokeConsentRequest', 1)
    name = _messages.StringField(2, required=True)