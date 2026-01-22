from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareDhcpIpConfig(_messages.Message):
    """Represents the network configuration required for the VMware user
  clusters with DHCP IP configurations.

  Fields:
    enabled: enabled is a flag to mark if DHCP IP allocation is used for
      VMware user clusters.
  """
    enabled = _messages.BooleanField(1)