from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaListPoliciesResponse(_messages.Message):
    """Response message for ListPolicies method.

  Fields:
    nextPageToken: A token to retrieve next page of results.
    policies: The list of policies.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('GoogleIamV3betaV3Policy', 2, repeated=True)