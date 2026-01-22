from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Screen(_messages.Message):
    """A Screen object.

  Fields:
    fileReference: File reference of the png file. Required.
    locale: Locale of the device that the screenshot was taken on. Required.
    model: Model of the device that the screenshot was taken on. Required.
    version: OS version of the device that the screenshot was taken on.
      Required.
  """
    fileReference = _messages.StringField(1)
    locale = _messages.StringField(2)
    model = _messages.StringField(3)
    version = _messages.StringField(4)