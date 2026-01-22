from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageRelease(_messages.Message):
    """ConfigImage represents an image release available to create a WbI

  Fields:
    imageName: Output only. The name of the image of the form workbench-
      instances-vYYYYmmdd--
    releaseName: Output only. The release of the image of the form m123
  """
    imageName = _messages.StringField(1)
    releaseName = _messages.StringField(2)