from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvelopeSignature(_messages.Message):
    """A EnvelopeSignature object.

  Fields:
    keyid: A string attribute.
    sig: A byte attribute.
  """
    keyid = _messages.StringField(1)
    sig = _messages.BytesField(2)