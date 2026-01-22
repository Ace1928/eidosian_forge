from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Color(_messages.Message):
    """Represents a color in the RGB color space.

  Fields:
    blue: The amount of blue in the color as a value in the interval [0, 1].
    green: The amount of green in the color as a value in the interval [0, 1].
    red: The amount of red in the color as a value in the interval [0, 1].
  """
    blue = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    green = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    red = _messages.FloatField(3, variant=_messages.Variant.FLOAT)