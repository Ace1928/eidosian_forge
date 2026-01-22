from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesResizeRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesResizeRequest object.

  Fields:
    resizeVolumeRequest: A ResizeVolumeRequest resource to be passed as the
      request body.
    volume: Required. Volume to resize.
  """
    resizeVolumeRequest = _messages.MessageField('ResizeVolumeRequest', 1)
    volume = _messages.StringField(2, required=True)