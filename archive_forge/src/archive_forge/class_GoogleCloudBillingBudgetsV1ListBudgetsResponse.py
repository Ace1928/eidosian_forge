from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1ListBudgetsResponse(_messages.Message):
    """Response for ListBudgets

  Fields:
    budgets: List of the budgets owned by the requested billing account.
    nextPageToken: If not empty, indicates that there may be more budgets that
      match the request; this value should be passed in a new
      `ListBudgetsRequest`.
  """
    budgets = _messages.MessageField('GoogleCloudBillingBudgetsV1Budget', 1, repeated=True)
    nextPageToken = _messages.StringField(2)