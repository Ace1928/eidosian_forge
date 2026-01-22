from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingbudgetsBillingAccountsBudgetsCreateRequest(_messages.Message):
    """A BillingbudgetsBillingAccountsBudgetsCreateRequest object.

  Fields:
    googleCloudBillingBudgetsV1Budget: A GoogleCloudBillingBudgetsV1Budget
      resource to be passed as the request body.
    parent: Required. The name of the billing account to create the budget in.
      Values are of the form `billingAccounts/{billingAccountId}`.
  """
    googleCloudBillingBudgetsV1Budget = _messages.MessageField('GoogleCloudBillingBudgetsV1Budget', 1)
    parent = _messages.StringField(2, required=True)