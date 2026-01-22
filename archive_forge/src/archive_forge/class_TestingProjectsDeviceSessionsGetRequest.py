from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsDeviceSessionsGetRequest(_messages.Message):
    """A TestingProjectsDeviceSessionsGetRequest object.

  Fields:
    name: Required. Name of the DeviceSession, e.g.
      "projects/{project_id}/deviceSessions/{session_id}"
  """
    name = _messages.StringField(1, required=True)