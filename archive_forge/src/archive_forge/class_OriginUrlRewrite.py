from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginUrlRewrite(_messages.Message):
    """The URL rewrite configuration for a given request handled by this
  origin.

  Fields:
    hostRewrite: Optional. Before forwarding the request to the selected
      origin, the request's `Host` header is replaced with the contents of
      `hostRewrite`. The host value must be between 1 and 255 characters.
  """
    hostRewrite = _messages.StringField(1)