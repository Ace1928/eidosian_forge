from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class XcodeVersion(_messages.Message):
    """An Xcode version that an iOS version is compatible with.

  Fields:
    tags: Tags for this Xcode version. Example: "default".
    version: The id for this version. Example: "9.2".
  """
    tags = _messages.StringField(1, repeated=True)
    version = _messages.StringField(2)