from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def GetNetworkInterfacesAlpha(args, client, holder, project, location, scope, skip_defaults):
    """Get network interfaces in compute Alpha API."""
    network_interface_args = filter(lambda flag: hasattr(args, flag), ['address', 'ipv6_network_tier', 'ipv6_public_ptr_domain', 'network', 'network_tier', 'no_address', 'no_public_dns', 'no_public_ptr', 'no_public_ptr_domain', 'private_network_ip', 'public_dns', 'public_ptr', 'public_ptr_domain', 'stack_type', 'subnet', 'ipv6_address', 'ipv6_prefix_length', 'internal_ipv6_address', 'internal_ipv6_prefix_length', 'external_ipv6_address', 'external_ipv6_prefix_length'])
    if skip_defaults and (not instance_utils.IsAnySpecified(args, *network_interface_args)):
        return []
    return [instances_utils.CreateNetworkInterfaceMessage(resources=holder.resources, compute_client=client, network=args.network, subnet=args.subnet, no_address=args.no_address, address=args.address, project=project, location=location, scope=scope, private_network_ip=getattr(args, 'private_network_ip', None), network_tier=getattr(args, 'network_tier', None), no_public_dns=getattr(args, 'no_public_dns', None), public_dns=getattr(args, 'public_dns', None), no_public_ptr=getattr(args, 'no_public_ptr', None), public_ptr=getattr(args, 'public_ptr', None), no_public_ptr_domain=getattr(args, 'no_public_ptr_domain', None), public_ptr_domain=getattr(args, 'public_ptr_domain', None), stack_type=getattr(args, 'stack_type', None), ipv6_network_tier=getattr(args, 'ipv6_network_tier', None), ipv6_public_ptr_domain=getattr(args, 'ipv6_public_ptr_domain', None), ipv6_address=getattr(args, 'ipv6_address', None), ipv6_prefix_length=getattr(args, 'ipv6_prefix_length', None), internal_ipv6_address=getattr(args, 'internal_ipv6_address', None), internal_ipv6_prefix_length=getattr(args, 'internal_ipv6_prefix_length', None), external_ipv6_address=getattr(args, 'external_ipv6_address', None), external_ipv6_prefix_length=getattr(args, 'external_ipv6_prefix_length', None))]