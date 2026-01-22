from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationEndpoint(_messages.Message):
    """ApplicationEndpoint represents a remote application endpoint.

  Fields:
    host: Required. Hostname or IP address of the remote application endpoint.
    port: Required. Port of the remote application endpoint.
  """
    host = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)