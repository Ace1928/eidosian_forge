from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesResetInstanceRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesResetInstanceRequest
  object.

  Fields:
    instance: Required. Name of the instance to reset.
    resetInstanceRequest: A ResetInstanceRequest resource to be passed as the
      request body.
  """
    instance = _messages.StringField(1, required=True)
    resetInstanceRequest = _messages.MessageField('ResetInstanceRequest', 2)