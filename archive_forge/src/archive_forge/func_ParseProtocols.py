from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseProtocols(self, volume, protocols):
    """Parses Protocols from a list of Protocol Enums into the given volume.

    Args:
      volume: The Cloud NetApp Volume message object
      protocols: A list of protocol enums

    Returns:
      Volume message populated with protocol values.

    """
    protocols_config = []
    for protocol in protocols:
        protocols_config.append(protocol)
    volume.protocols = protocols_config