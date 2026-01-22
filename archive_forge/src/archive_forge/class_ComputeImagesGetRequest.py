from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeImagesGetRequest(_messages.Message):
    """A ComputeImagesGetRequest object.

  Fields:
    image: Name of the image resource to return.
    project: Project ID for this request.
  """
    image = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)