from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChipCoordinateRangeGenerator(_messages.Message):
    """Compactly represents a list of Chip coordinates as the cross-product of
  each term. For example: * slice_coordinates: { 0, 1, 2, 3 } * x_coordinates:
  { 0 } * y_coordinates: { 0, 1 } Represents all the chips in the first column
  (x) and first 2 rows (y) of the first 4 slices.

  Fields:
    sliceCoordinates: Slice coordinates for chip coordinate range
    xCoordinates: X coordinates for chip coordinate range.
    yCoordinates: Y coordinates for chip coordinate range.
    zCoordinates: If specifying 2D coordinates, z_coordinate may be omitted.
  """
    sliceCoordinates = _messages.MessageField('Range', 1)
    xCoordinates = _messages.MessageField('Range', 2)
    yCoordinates = _messages.MessageField('Range', 3)
    zCoordinates = _messages.MessageField('Range', 4)