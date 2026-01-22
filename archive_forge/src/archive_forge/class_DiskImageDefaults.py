from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskImageDefaults(_messages.Message):
    """Contains details about the image source used to create the disk.

  Fields:
    sourceImage: Required. The Image resource used when creating the disk.
  """
    sourceImage = _messages.StringField(1)