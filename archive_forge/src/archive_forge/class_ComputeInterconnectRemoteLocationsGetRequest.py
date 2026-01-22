from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInterconnectRemoteLocationsGetRequest(_messages.Message):
    """A ComputeInterconnectRemoteLocationsGetRequest object.

  Fields:
    interconnectRemoteLocation: Name of the interconnect remote location to
      return.
    project: Project ID for this request.
  """
    interconnectRemoteLocation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)