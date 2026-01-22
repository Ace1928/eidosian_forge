from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsObjectsGetRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsObjectsGetRequest object.

  Fields:
    name: Required. The name of the stream object resource to get.
  """
    name = _messages.StringField(1, required=True)