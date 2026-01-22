from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateHealthMonitor(neutronV20.CreateCommand):
    """LBaaS v2 Create a healthmonitor."""
    resource = 'healthmonitor'
    shadow_resource = 'lbaas_healthmonitor'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--type', required=True, choices=['PING', 'TCP', 'HTTP', 'HTTPS'], help=_('One of the predefined health monitor types.'))
        parser.add_argument('--pool', required=True, help=_('ID or name of the pool that this healthmonitor will monitor.'))

    def args2body(self, parsed_args):
        pool_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'pool', parsed_args.pool, cmd_resource='lbaas_pool')
        body = {'admin_state_up': parsed_args.admin_state, 'type': parsed_args.type, 'pool_id': pool_id}
        neutronV20.update_dict(parsed_args, body, ['tenant_id'])
        _parse_common_args(body, parsed_args)
        return {self.resource: body}