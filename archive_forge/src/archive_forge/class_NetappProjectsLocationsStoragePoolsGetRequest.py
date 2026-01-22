from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsStoragePoolsGetRequest(_messages.Message):
    """A NetappProjectsLocationsStoragePoolsGetRequest object.

  Fields:
    name: Required. Name of the storage pool
  """
    name = _messages.StringField(1, required=True)