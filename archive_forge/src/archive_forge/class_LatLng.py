from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatLng(_messages.Message):
    """An object that represents a latitude/longitude pair. This is expressed
  as a pair of doubles to represent degrees latitude and degrees longitude.
  Unless specified otherwise, this object must conform to the WGS84 standard.
  Values must be within normalized ranges.

  Fields:
    latitude: The latitude in degrees. It must be in the range [-90.0, +90.0].
    longitude: The longitude in degrees. It must be in the range [-180.0,
      +180.0].
  """
    latitude = _messages.FloatField(1)
    longitude = _messages.FloatField(2)