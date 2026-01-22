from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsAccountsCreateRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsAccountsCreateRequest
  object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1Account: A
      GoogleCloudCommerceConsumerProcurementV1alpha1Account resource to be
      passed as the request body.
    parent: Required. The parent resource of this account. This field is of
      the form "/". Currently supported type: 'billingAccounts/{billing-
      account-id}'
  """
    googleCloudCommerceConsumerProcurementV1alpha1Account = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Account', 1)
    parent = _messages.StringField(2, required=True)