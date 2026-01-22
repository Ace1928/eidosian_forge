from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersBalanceCreditRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersBalanceCreditRequest object.

  Fields:
    googleCloudApigeeV1CreditDeveloperBalanceRequest: A
      GoogleCloudApigeeV1CreditDeveloperBalanceRequest resource to be passed
      as the request body.
    name: Required. Account balance for the developer. Use the following
      structure in your request:
      `organizations/{org}/developers/{developer}/balance`
  """
    googleCloudApigeeV1CreditDeveloperBalanceRequest = _messages.MessageField('GoogleCloudApigeeV1CreditDeveloperBalanceRequest', 1)
    name = _messages.StringField(2, required=True)