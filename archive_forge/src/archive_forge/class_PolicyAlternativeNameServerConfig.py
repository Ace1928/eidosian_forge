from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PolicyAlternativeNameServerConfig(_messages.Message):
    """A PolicyAlternativeNameServerConfig object.

  Fields:
    kind: A string attribute.
    targetNameServers: Sets an alternative name server for the associated
      networks. When specified, all DNS queries are forwarded to a name server
      that you choose. Names such as .internal are not available when an
      alternative name server is specified.
  """
    kind = _messages.StringField(1, default='dns#policyAlternativeNameServerConfig')
    targetNameServers = _messages.MessageField('PolicyAlternativeNameServerConfigTargetNameServer', 2, repeated=True)