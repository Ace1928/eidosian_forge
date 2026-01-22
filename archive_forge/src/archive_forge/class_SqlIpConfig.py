from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlIpConfig(_messages.Message):
    """IP Management configuration.

  Fields:
    authorizedNetworks: The list of external networks that are allowed to
      connect to the instance using the IP. See
      https://en.wikipedia.org/wiki/CIDR_notation#CIDR_notation, also known as
      'slash' notation (e.g. `192.168.100.0/24`).
    enableIpv4: Whether the instance is assigned a public IP address or not.
    privateNetwork: The resource link for the VPC network from which the Cloud
      SQL instance is accessible for private IP. For example,
      `/projects/myProject/global/networks/default`. This setting can be
      updated, but it cannot be removed after it is set.
    requireSsl: Whether SSL connections over IP should be enforced or not.
  """
    authorizedNetworks = _messages.MessageField('SqlAclEntry', 1, repeated=True)
    enableIpv4 = _messages.BooleanField(2)
    privateNetwork = _messages.StringField(3)
    requireSsl = _messages.BooleanField(4)