from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsActiveDirectoriesGetRequest(_messages.Message):
    """A NetappProjectsLocationsActiveDirectoriesGetRequest object.

  Fields:
    name: Required. Name of the active directory.
  """
    name = _messages.StringField(1, required=True)