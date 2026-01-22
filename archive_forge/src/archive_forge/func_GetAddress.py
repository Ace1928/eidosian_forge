from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags
import ipaddr
from six.moves import zip  # pylint: disable=redefined-builtin
def GetAddress(self, messages, args, address, address_ref, resource_parser):
    """Get and validate address setting.

    Retrieve address resource from input arguments and validate the address
    configuration for both external/internal IP address reservation/promotion.

    Args:
      messages: The client message proto includes all required GCE API fields.
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.
      address: Address object.
      address_ref: Reference of the address.
      resource_parser: A resource parser used to parse resource name into url.

    Returns:
      An address resource proto message.

    Raises:
      ConflictingArgumentsException: If both network and subnetwork fields are
      set.
      MinimumArgumentException: Missing network or subnetwork with purpose
      field.
      InvalidArgumentException: The input argument is not set correctly or
      unable to be parsed.
      RequiredArgumentException: The required argument is missing from user
      input.
    """
    network_tier = self.ConstructNetworkTier(messages, args)
    if args.ip_version or (address is None and address_ref.Collection() == 'compute.globalAddresses'):
        ip_version = messages.Address.IpVersionValueValuesEnum(args.ip_version or 'IPV4')
    else:
        ip_version = None
    if args.subnet and args.network:
        raise exceptions.ConflictingArgumentsException('--network', '--subnet')
    purpose = None
    if args.purpose and (not args.network) and (not args.subnet):
        raise exceptions.MinimumArgumentException(['--network', '--subnet'], ' if --purpose is specified')
    if args.subnet:
        if address_ref.Collection() == 'compute.globalAddresses':
            raise exceptions.BadArgumentException('--subnet', '[--subnet] may not be specified for global addresses.')
        if not args.subnet_region:
            args.subnet_region = address_ref.region
        subnetwork_url = flags.SubnetworkArgument().ResolveAsResource(args, resource_parser).SelfLink()
        if not args.endpoint_type:
            purpose = messages.Address.PurposeValueValuesEnum(args.purpose or 'GCE_ENDPOINT')
            self.CheckPurposeInSubnetwork(messages, purpose)
    else:
        subnetwork_url = None
    network_url = None
    if args.network:
        purpose = messages.Address.PurposeValueValuesEnum(args.purpose or 'VPC_PEERING')
        network_url = flags.NetworkArgument().ResolveAsResource(args, resource_parser).SelfLink()
        if purpose != messages.Address.PurposeValueValuesEnum.IPSEC_INTERCONNECT:
            if address_ref.Collection() == 'compute.addresses':
                raise exceptions.InvalidArgumentException('--network', 'network may not be specified for regional addresses.')
            supported_purposes = {'VPC_PEERING': messages.Address.PurposeValueValuesEnum.VPC_PEERING}
            if self._support_psc_google_apis:
                supported_purposes['PRIVATE_SERVICE_CONNECT'] = messages.Address.PurposeValueValuesEnum.PRIVATE_SERVICE_CONNECT
            if purpose not in supported_purposes.values():
                raise exceptions.InvalidArgumentException('--purpose', 'must be {} for global internal addresses.'.format(' or '.join(supported_purposes.keys())))
    ipv6_endpoint_type = None
    if args.endpoint_type:
        ipv6_endpoint_type = messages.Address.Ipv6EndpointTypeValueValuesEnum(args.endpoint_type)
    address_type = None
    if args.endpoint_type:
        address_type = messages.Address.AddressTypeValueValuesEnum.EXTERNAL
    elif subnetwork_url or network_url:
        address_type = messages.Address.AddressTypeValueValuesEnum.INTERNAL
    if address is not None:
        try:
            ip_address = ipaddr.IPAddress(address)
        except ValueError:
            raise exceptions.InvalidArgumentException('--addresses', 'Invalid IP address {e}'.format(e=address))
    if args.prefix_length:
        if address and (not address_type):
            address_type = messages.Address.AddressTypeValueValuesEnum.EXTERNAL
        elif (address is None or ip_address.version != 6) and purpose != messages.Address.PurposeValueValuesEnum.VPC_PEERING and (purpose != messages.Address.PurposeValueValuesEnum.IPSEC_INTERCONNECT):
            raise exceptions.InvalidArgumentException('--prefix-length', 'can only be used with [--purpose VPC_PEERING/IPSEC_INTERCONNECT] or Internal/External IPv6 reservation. Found {e}'.format(e=purpose))
    if not args.prefix_length:
        if purpose == messages.Address.PurposeValueValuesEnum.VPC_PEERING:
            raise exceptions.RequiredArgumentException('--prefix-length', 'prefix length is needed for reserving VPC peering IP ranges.')
        if purpose == messages.Address.PurposeValueValuesEnum.IPSEC_INTERCONNECT:
            raise exceptions.RequiredArgumentException('--prefix-length', 'prefix length is needed for reserving IP ranges for HA VPN over Cloud Interconnect.')
    return messages.Address(address=address, prefixLength=args.prefix_length, description=args.description, networkTier=network_tier, ipVersion=ip_version, name=address_ref.Name(), addressType=address_type, purpose=purpose, subnetwork=subnetwork_url, network=network_url, ipv6EndpointType=ipv6_endpoint_type)