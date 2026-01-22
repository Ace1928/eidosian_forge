from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminNetworkConfig(_messages.Message):
    """BareMetalAdminNetworkConfig specifies the cluster network configuration.

  Fields:
    islandModeCidr: Configuration for Island mode CIDR.
  """
    islandModeCidr = _messages.MessageField('BareMetalAdminIslandModeCidrConfig', 1)