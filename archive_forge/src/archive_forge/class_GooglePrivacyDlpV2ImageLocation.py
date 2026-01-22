from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ImageLocation(_messages.Message):
    """Location of the finding within an image.

  Fields:
    boundingBoxes: Bounding boxes locating the pixels within the image
      containing the finding.
  """
    boundingBoxes = _messages.MessageField('GooglePrivacyDlpV2BoundingBox', 1, repeated=True)