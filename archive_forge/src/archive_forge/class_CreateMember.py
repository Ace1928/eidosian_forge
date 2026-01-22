from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateMember(neutronV20.CreateCommand):
    """LBaaS v2 Create a member."""
    resource = 'member'
    shadow_resource = 'lbaas_member'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--subnet', required=True, help=_('Subnet ID or name for the member.'))
        parser.add_argument('--address', required=True, help=_('IP address of the pool member in the pool.'))
        parser.add_argument('--protocol-port', required=True, help=_('Port on which the pool member listens for requests or connections.'))
        parser.add_argument('pool', metavar='POOL', help=_('ID or name of the pool that this member belongs to.'))

    def args2body(self, parsed_args):
        self.parent_id = _get_pool_id(self.get_client(), parsed_args.pool)
        _subnet_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', parsed_args.subnet)
        body = {'subnet_id': _subnet_id, 'admin_state_up': parsed_args.admin_state, 'protocol_port': parsed_args.protocol_port, 'address': parsed_args.address}
        neutronV20.update_dict(parsed_args, body, ['subnet_id', 'tenant_id'])
        _parse_common_args(body, parsed_args)
        return {self.resource: body}