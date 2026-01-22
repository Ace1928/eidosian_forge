from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsDeviceSessionsCreateRequest(_messages.Message):
    """A TestingProjectsDeviceSessionsCreateRequest object.

  Fields:
    deviceSession: A DeviceSession resource to be passed as the request body.
    parent: Required. The Compute Engine project under which this device will
      be allocated. "projects/{project_id}"
  """
    deviceSession = _messages.MessageField('DeviceSession', 1)
    parent = _messages.StringField(2, required=True)