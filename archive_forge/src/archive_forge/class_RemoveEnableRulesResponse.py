from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoveEnableRulesResponse(_messages.Message):
    """The response message of `RemoveEnableRules` method.

  Fields:
    parent: The parent consumer policy. It can be
      `projects/12345/consumerPolicies/default`, or
      `folders/12345/consumerPolicies/default`, or
      `organizations/12345/consumerPolicies/default`.
    removedValues: The values removed from the parent consumer policy.
  """
    parent = _messages.StringField(1)
    removedValues = _messages.StringField(2, repeated=True)