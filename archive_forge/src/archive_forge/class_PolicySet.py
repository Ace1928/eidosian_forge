from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicySet(_messages.Message):
    """PolicySet representation.

  Fields:
    description: Optional. Description of the Policy set.
    policies: Required. List of policies.
    policySetId: Required. ID of the Policy set.
  """
    description = _messages.StringField(1)
    policies = _messages.MessageField('Policy', 2, repeated=True)
    policySetId = _messages.StringField(3)