from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.compute.routers import router_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateBgpPeerBeta(UpdateBgpPeer):
    """Update a BGP peer on a Compute Engine router."""
    ROUTER_ARG = None

    @classmethod
    def Args(cls, parser):
        cls._Args(parser, enable_ipv6_bgp=True)

    def Run(self, args):
        """Runs the command.

    Args:
      args: contains arguments passed to the command.

    Returns:
      The result of patching the router updating the bgp peer with the
      information provided in the arguments.
    """
        return self._Run(args, support_bfd_mode=False, enable_ipv6_bgp=True)