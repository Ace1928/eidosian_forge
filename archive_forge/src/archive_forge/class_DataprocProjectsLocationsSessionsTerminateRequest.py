from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsTerminateRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsTerminateRequest object.

  Fields:
    name: Required. The name of the session resource to terminate.
    terminateSessionRequest: A TerminateSessionRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    terminateSessionRequest = _messages.MessageField('TerminateSessionRequest', 2)