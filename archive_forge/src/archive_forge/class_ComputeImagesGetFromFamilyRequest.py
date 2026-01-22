from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeImagesGetFromFamilyRequest(_messages.Message):
    """A ComputeImagesGetFromFamilyRequest object.

  Fields:
    family: Name of the image family to search for.
    project: The image project that the image belongs to. For example, to get
      a CentOS image, specify centos-cloud as the image project.
  """
    family = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)