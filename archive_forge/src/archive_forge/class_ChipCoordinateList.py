from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChipCoordinateList(_messages.Message):
    """Represents a list of individually defined chip coordinates.

  Fields:
    coordinates: List of chip coordinates.
  """
    coordinates = _messages.MessageField('ChipCoordinate', 1, repeated=True)