from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.vpn_gateways import vpn_gateways_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.compute.vpn_gateways import flags
def _mapInterconnectAttachments(self, args, resources, region, project):
    """Returns dict {interfaceId : interconnectAttachmentUrl} based on initial order of names in input interconnectAttachmentName and region and project of VPN Gateway.

    Args:
      args: Namespace, argparse.Namespace.
      resources: Generates resource references.
      region: VPN Gateway region.
      project: VPN Gateway project.
    """
    attachment_refs = args.interconnect_attachments
    result = {0: flags.GetInterconnectAttachmentRef(resources, attachment_refs[0], region, project).SelfLink(), 1: flags.GetInterconnectAttachmentRef(resources, attachment_refs[1], region, project).SelfLink()}
    return result