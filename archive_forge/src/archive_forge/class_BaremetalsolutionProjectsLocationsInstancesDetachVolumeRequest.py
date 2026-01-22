from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesDetachVolumeRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesDetachVolumeRequest object.

  Fields:
    detachVolumeRequest: A DetachVolumeRequest resource to be passed as the
      request body.
    instance: Required. Name of the instance.
  """
    detachVolumeRequest = _messages.MessageField('DetachVolumeRequest', 1)
    instance = _messages.StringField(2, required=True)