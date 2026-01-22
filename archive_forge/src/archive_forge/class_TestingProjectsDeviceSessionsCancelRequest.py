from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsDeviceSessionsCancelRequest(_messages.Message):
    """A TestingProjectsDeviceSessionsCancelRequest object.

  Fields:
    cancelDeviceSessionRequest: A CancelDeviceSessionRequest resource to be
      passed as the request body.
    name: Required. Name of the DeviceSession, e.g.
      "projects/{project_id}/deviceSessions/{session_id}"
  """
    cancelDeviceSessionRequest = _messages.MessageField('CancelDeviceSessionRequest', 1)
    name = _messages.StringField(2, required=True)