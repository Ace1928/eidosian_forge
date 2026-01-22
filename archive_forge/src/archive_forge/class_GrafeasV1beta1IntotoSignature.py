from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1IntotoSignature(_messages.Message):
    """A signature object consists of the KeyID used and the signature itself.

  Fields:
    keyid: A string attribute.
    sig: A string attribute.
  """
    keyid = _messages.StringField(1)
    sig = _messages.StringField(2)