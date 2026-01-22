from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingbudgetsBillingAccountsBudgetsGetRequest(_messages.Message):
    """A BillingbudgetsBillingAccountsBudgetsGetRequest object.

  Fields:
    name: Required. Name of budget to get. Values are of the form
      `billingAccounts/{billingAccountId}/budgets/{budgetId}`.
  """
    name = _messages.StringField(1, required=True)