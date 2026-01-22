from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcaccessProjectsLocationsConnectorsHeartbeatRequest(_messages.Message):
    """A VpcaccessProjectsLocationsConnectorsHeartbeatRequest object.

  Fields:
    heartbeatConnectorRequest: A HeartbeatConnectorRequest resource to be
      passed as the request body.
    name: Required.
  """
    heartbeatConnectorRequest = _messages.MessageField('HeartbeatConnectorRequest', 1)
    name = _messages.StringField(2, required=True)