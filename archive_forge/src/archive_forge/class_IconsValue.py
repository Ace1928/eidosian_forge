from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IconsValue(_messages.Message):
    """Links to 16x16 and 32x32 icons representing the API.

      Fields:
        x16: The url of the 16x16 icon.
        x32: The url of the 32x32 icon.
      """
    x16 = _messages.StringField(1)
    x32 = _messages.StringField(2)