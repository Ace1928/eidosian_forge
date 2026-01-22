from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeStoragePoolsGetRequest(_messages.Message):
    """A ComputeStoragePoolsGetRequest object.

  Fields:
    project: Project ID for this request.
    storagePool: Name of the storage pool to return.
    zone: The name of the zone for this request.
  """
    project = _messages.StringField(1, required=True)
    storagePool = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)