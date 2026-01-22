from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkMountPoint(_messages.Message):
    """Mount point for a network.

  Fields:
    defaultGateway: Network should be a default gateway.
    instance: Instance to attach network to.
    ipAddress: Ip address of the server.
    logicalInterface: Logical interface to detach from.
  """
    defaultGateway = _messages.BooleanField(1)
    instance = _messages.StringField(2)
    ipAddress = _messages.StringField(3)
    logicalInterface = _messages.StringField(4)