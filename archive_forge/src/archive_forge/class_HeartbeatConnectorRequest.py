from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeartbeatConnectorRequest(_messages.Message):
    """Heartbeat requests come in from each connector VM to report their IP and
  serving state.

  Fields:
    heartbeatTime: Required. When this request was sent.
    ipAddress: Required. The IP address of the VM.
    lameduck: If the VM is in lameduck mode, meaning that it is in the process
      of shutting down and should not be used for new connections.
    projectNumber: The host project number for the VPC network that the VM is
      programmed to talk to. In shared VPC this may differ from the project
      number that the Connector and Serverless app attached to it belong to.
  """
    heartbeatTime = _messages.StringField(1)
    ipAddress = _messages.StringField(2)
    lameduck = _messages.BooleanField(3)
    projectNumber = _messages.StringField(4)