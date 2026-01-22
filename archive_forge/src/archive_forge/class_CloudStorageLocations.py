from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageLocations(_messages.Message):
    """Collection of Cloud Storage locations. Next ID: 2

  Fields:
    locations: A string attribute.
  """
    locations = _messages.StringField(1, repeated=True)