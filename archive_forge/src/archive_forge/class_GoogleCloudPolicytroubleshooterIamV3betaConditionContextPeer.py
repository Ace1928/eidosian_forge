from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaConditionContextPeer(_messages.Message):
    """This message defines attributes for a node that handles a network
  request. The node can be either a service or an application that sends,
  forwards, or receives the request. Service peers should fill in `principal`
  and `labels` as appropriate.

  Fields:
    ip: The IPv4 or IPv6 address of the peer.
    port: The network port of the peer.
  """
    ip = _messages.StringField(1)
    port = _messages.IntegerField(2)