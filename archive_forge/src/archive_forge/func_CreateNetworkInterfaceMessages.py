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
def CreateNetworkInterfaceMessages(resources, compute_client, network_interface_arg, project, location, scope, network_interface_json=None, support_internal_ipv6_reservation=False):
    """Create network interface messages.

  Args:
    resources: generates resource references.
    compute_client: creates resources.
    network_interface_arg: CLI argument specifying network interfaces.
    project: project of the instance that will own the generated network
      interfaces.
    location: Location of the instance that will own the new network interfaces.
    scope: Location type of the instance that will own the new network
      interfaces.
    network_interface_json: CLI argument value specifying network interfaces in
      a JSON string directly in the command or in a file.
    support_internal_ipv6_reservation: The flag indicates whether internal IPv6
      reservation is supported.

  Returns:
    list, items are NetworkInterfaceMessages.
  """
    result = []
    if network_interface_arg:
        for interface in network_interface_arg:
            address = interface.get('address', None)
            no_address = 'no-address' in interface
            network_tier = interface.get('network-tier', None)
            internal_ipv6_address = None
            internal_ipv6_prefix_length = None
            if support_internal_ipv6_reservation:
                internal_ipv6_address = interface.get('internal-ipv6-address', None)
                internal_ipv6_prefix_length = interface.get('internal-ipv6-prefix-length', None)
            result.append(instances_utils.CreateNetworkInterfaceMessage(resources=resources, compute_client=compute_client, network=interface.get('network', None), subnet=interface.get('subnet', None), private_network_ip=interface.get('private-network-ip', None), nic_type=interface.get('nic-type', None), no_address=no_address, address=address, project=project, location=location, scope=scope, alias_ip_ranges_string=interface.get('aliases', None), network_tier=network_tier, stack_type=interface.get('stack-type', None), ipv6_network_tier=interface.get('ipv6-network-tier', None), ipv6_public_ptr_domain=interface.get('ipv6-public-ptr-domain', None), queue_count=interface.get('queue-count', None), network_attachment=interface.get('network-attachment', None), internal_ipv6_address=internal_ipv6_address, internal_ipv6_prefix_length=internal_ipv6_prefix_length, external_ipv6_address=interface.get('external-ipv6-address', None), external_ipv6_prefix_length=interface.get('external-ipv6-prefix-length', None), vlan=interface.get('vlan', None), igmp_query=interface.get('igmp-query', None)))
    elif network_interface_json is not None:
        network_interfaces = yaml.load(network_interface_json)
        if not network_interfaces:
            return result
        for interface in network_interfaces:
            if not interface:
                continue
            network_interface = messages_util.DictToMessageWithErrorCheck(interface, compute_client.messages.NetworkInterface)
            result.append(network_interface)
    return result