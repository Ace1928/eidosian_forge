from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoundToScale(_messages.Message):
    """This allows the data to change scale, for example if the source is 2
  digits after the decimal point, specify round to scale value = 2. If for
  example the value needs to be converted to an integer, use round to scale
  value = 0.

  Fields:
    scale: Required. Scale value to be used
  """
    scale = _messages.IntegerField(1, variant=_messages.Variant.INT32)