from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRuleReference(_messages.Message):
    """A ForwardingRuleReference object.

  Fields:
    forwardingRule: A string attribute.
  """
    forwardingRule = _messages.StringField(1)