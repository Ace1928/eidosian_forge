from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareHostIp(_messages.Message):
    """Represents VMware user cluster node's network configuration.

  Fields:
    hostname: Hostname of the machine. VM's name will be used if this field is
      empty.
    ip: IP could be an IP address (like 1.2.3.4) or a CIDR (like 1.2.3.0/24).
  """
    hostname = _messages.StringField(1)
    ip = _messages.StringField(2)