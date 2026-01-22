from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageFamilyView(_messages.Message):
    """A ImageFamilyView object.

  Fields:
    image: The latest image that is part of the specified image family in the
      requested location, and that is not deprecated.
  """
    image = _messages.MessageField('Image', 1)