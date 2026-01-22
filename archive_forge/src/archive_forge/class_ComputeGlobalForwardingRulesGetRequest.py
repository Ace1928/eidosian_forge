from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeGlobalForwardingRulesGetRequest(_messages.Message):
    """A ComputeGlobalForwardingRulesGetRequest object.

  Fields:
    forwardingRule: Name of the ForwardingRule resource to return.
    project: Project ID for this request.
  """
    forwardingRule = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)