from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.interconnects.attachments import (
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers import flags as router_flags
from googlecloudsdk.command_lib.compute.vpn_tunnels import (
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AddInterfaceAlpha(AddInterfaceBeta):
    """Add an interface to a Compute Engine router.

  *{command}* is used to add an interface to a Compute Engine
  router.
  """
    pass