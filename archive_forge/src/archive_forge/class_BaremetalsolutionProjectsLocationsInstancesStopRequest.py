from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesStopRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesStopRequest object.

  Fields:
    name: Required. Name of the resource.
    stopInstanceRequest: A StopInstanceRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    stopInstanceRequest = _messages.MessageField('StopInstanceRequest', 2)