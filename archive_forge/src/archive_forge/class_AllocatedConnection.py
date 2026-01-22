from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocatedConnection(_messages.Message):
    """Allocated connection of the AppGateway.

  Fields:
    ingressPort: Required. The ingress port of an allocated connection
    pscUri: Required. The PSC uri of an allocated connection
  """
    ingressPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pscUri = _messages.StringField(2)