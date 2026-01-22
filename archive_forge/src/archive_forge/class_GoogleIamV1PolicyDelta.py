from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1PolicyDelta(_messages.Message):
    """The difference delta between two policies.

  Fields:
    bindingDeltas: The delta for Bindings between two policies.
  """
    bindingDeltas = _messages.MessageField('GoogleIamV1BindingDelta', 1, repeated=True)