from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2ServerKeyRestrictions(_messages.Message):
    """The IP addresses of callers that are allowed to use the key.

  Fields:
    allowedIps: A list of the caller IP addresses that are allowed to make API
      calls with this key.
  """
    allowedIps = _messages.StringField(1, repeated=True)