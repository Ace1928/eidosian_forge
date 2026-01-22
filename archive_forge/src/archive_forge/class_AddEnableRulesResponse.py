from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddEnableRulesResponse(_messages.Message):
    """The response message of `AddEnableRules` method.

  Fields:
    addedValues: The values added to the parent consumer policy.
    parent: The parent consumer policy. It can be
      `projects/12345/consumerPolicies/default`, or
      `folders/12345/consumerPolicies/default`, or
      `organizations/12345/consumerPolicies/default`.
  """
    addedValues = _messages.StringField(1, repeated=True)
    parent = _messages.StringField(2)