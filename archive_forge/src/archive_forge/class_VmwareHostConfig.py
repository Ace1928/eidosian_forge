from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareHostConfig(_messages.Message):
    """Represents the common parameters for all the hosts irrespective of their
  IP address.

  Fields:
    dnsSearchDomains: DNS search domains.
    dnsServers: DNS servers.
    ntpServers: NTP servers.
  """
    dnsSearchDomains = _messages.StringField(1, repeated=True)
    dnsServers = _messages.StringField(2, repeated=True)
    ntpServers = _messages.StringField(3, repeated=True)