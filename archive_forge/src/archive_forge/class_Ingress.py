from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Ingress(_messages.Message):
    """Settings of how to connect to the ClientGateway. One of the following
  options should be set.

  Fields:
    config: The basic ingress config for ClientGateways.
  """
    config = _messages.MessageField('Config', 1)