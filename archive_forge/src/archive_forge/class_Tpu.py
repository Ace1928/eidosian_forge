from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Tpu(_messages.Message):
    """Details of the TPU resource(s) being requested.

  Fields:
    nodeSpec: Optional. The TPU node(s) being requested.
  """
    nodeSpec = _messages.MessageField('NodeSpec', 1, repeated=True)