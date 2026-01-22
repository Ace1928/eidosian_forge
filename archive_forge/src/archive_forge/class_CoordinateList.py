from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CoordinateList(_messages.Message):
    """Defines a list of related `src` and/or `dst` coordinates in the traffic
  matrix.

  Fields:
    chipCoordinate: A list of individually defined chip coordinates.
    chipCoordinateRangeGenerator: A list of chip coordinates represented by
      the provided range generator.
  """
    chipCoordinate = _messages.MessageField('ChipCoordinateList', 1)
    chipCoordinateRangeGenerator = _messages.MessageField('ChipCoordinateRangeGenerator', 2)