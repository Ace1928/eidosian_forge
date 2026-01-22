from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsOsImagesGetRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsOsImagesGetRequest object.

  Fields:
    name: Required. Name of the OS image.
  """
    name = _messages.StringField(1, required=True)