from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityActionHttpHeader(_messages.Message):
    """An HTTP header.

  Fields:
    name: The header name to be sent to the target.
    value: The header value to be sent to the target.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)