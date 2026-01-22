from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageAnnotation(_messages.Message):
    """Image annotation.

  Fields:
    boundingPolys: The list of polygons outlining the sensitive regions in the
      image.
    frameIndex: 0-based index of the image frame. For example, an image frame
      in a DICOM instance.
  """
    boundingPolys = _messages.MessageField('BoundingPoly', 1, repeated=True)
    frameIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)