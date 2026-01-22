from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyInfo(_messages.Message):
    """The IAM policy and its attached resource.

  Fields:
    attachedResource: The full resource name the policy is directly attached
      to.
    policy: The IAM policy that's directly attached to the attached_resource.
  """
    attachedResource = _messages.StringField(1)
    policy = _messages.MessageField('Policy', 2)