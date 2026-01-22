from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatLongRect(_messages.Message):
    """Rectangle determined by min and max `LatLng` pairs.

  Fields:
    maxLatLng: Max lat/long pair.
    minLatLng: Min lat/long pair.
  """
    maxLatLng = _messages.MessageField('LatLng', 1)
    minLatLng = _messages.MessageField('LatLng', 2)