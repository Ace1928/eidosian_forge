import logging
from operator import xor
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class CreateShareNetworkSubnet(command.ShowOne):
    """Create a share network subnet."""
    _description = _('Create a share network subnet')

    def get_parser(self, prog_name):
        parser = super(CreateShareNetworkSubnet, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', help=_('Share network name or ID.'))
        parser.add_argument('--neutron-net-id', metavar='<neutron-net-id>', default=None, help=_("Neutron network ID. Used to set up network for share servers (optional). Should be defined together with '--neutron-subnet-id'."))
        parser.add_argument('--neutron-subnet-id', metavar='<neutron-subnet-id>', default=None, help=_("Neutron subnet ID. Used to set up network for share servers (optional). Should be defined together with '--neutron-net-id' to which this subnet belongs to. "))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', default=None, help=_('Optional availability zone that the subnet is available within (Default=None). If None, the subnet will be considered as being available across all availability zones.'))
        parser.add_argument('--check-only', default=False, action='store_true', help=_('Run a dry-run of a share network subnet create. Available only for microversion >= 2.70.'))
        parser.add_argument('--restart-check', default=False, action='store_true', help=_('Restart a dry-run of a share network subnet create. Helpful when check results are stale. Available only for microversion >= 2.70.'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this share network subnet (repeat option to set multiple properties). Available only for microversion >= 2.78.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if parsed_args.check_only and share_client.api_version < api_versions.APIVersion('2.70'):
            raise exceptions.CommandError('Check only can be specified only with manila API version >= 2.70.')
        if parsed_args.restart_check and share_client.api_version < api_versions.APIVersion('2.70'):
            raise exceptions.CommandError('Restart check can be specified only with manila API version >= 2.70.')
        if parsed_args.property and share_client.api_version < api_versions.APIVersion('2.78'):
            raise exceptions.CommandError('Property can be specified only with manila API version >= 2.78.')
        if xor(bool(parsed_args.neutron_net_id), bool(parsed_args.neutron_subnet_id)):
            raise exceptions.CommandError('Both neutron_net_id and neutron_subnet_id should be specified. Alternatively, neither of them should be specified.')
        share_network_id = oscutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        if parsed_args.check_only or parsed_args.restart_check:
            if parsed_args.property:
                raise exceptions.CommandError('Property cannot be specified with check operation.')
            subnet_create_check = share_client.share_networks.share_network_subnet_create_check(neutron_net_id=parsed_args.neutron_net_id, neutron_subnet_id=parsed_args.neutron_subnet_id, availability_zone=parsed_args.availability_zone, reset_operation=parsed_args.restart_check, share_network_id=share_network_id)
            subnet_data = subnet_create_check[1]
        else:
            share_network_subnet = share_client.share_network_subnets.create(neutron_net_id=parsed_args.neutron_net_id, neutron_subnet_id=parsed_args.neutron_subnet_id, availability_zone=parsed_args.availability_zone, share_network_id=share_network_id, metadata=parsed_args.property)
            subnet_data = share_network_subnet._info
        return self.dict2columns(subnet_data)