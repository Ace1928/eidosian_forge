from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMapTestHeader(_messages.Message):
    """HTTP headers used in UrlMapTests.

  Fields:
    name: Header name.
    value: Header value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)