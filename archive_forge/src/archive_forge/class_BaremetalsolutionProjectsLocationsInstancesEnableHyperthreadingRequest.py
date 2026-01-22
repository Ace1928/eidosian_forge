from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesEnableHyperthreadingRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesEnableHyperthreadingRequest
  object.

  Fields:
    enableHyperthreadingRequest: A EnableHyperthreadingRequest resource to be
      passed as the request body.
    name: Required. The `name` field is used to identify the instance. Format:
      projects/{project}/locations/{location}/instances/{instance}
  """
    enableHyperthreadingRequest = _messages.MessageField('EnableHyperthreadingRequest', 1)
    name = _messages.StringField(2, required=True)