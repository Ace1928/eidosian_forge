from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.packet_mirrorings import client
from googlecloudsdk.command_lib.compute.packet_mirrorings import flags
from googlecloudsdk.command_lib.compute.packet_mirrorings import utils
def _MakeMirroredSubnetInfo(subnet):
    return messages.PacketMirroringMirroredResourceInfoSubnetInfo(url=utils.ResolveSubnetURI(pm_ref.project, pm_ref.region, subnet, holder.resources))