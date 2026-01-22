from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class DisassociateHealthMonitor(neutronV20.NeutronCommand):
    """Remove a mapping from a health monitor to a pool."""
    resource = 'health_monitor'

    def get_parser(self, prog_name):
        parser = super(DisassociateHealthMonitor, self).get_parser(prog_name)
        parser.add_argument('health_monitor_id', metavar='HEALTH_MONITOR_ID', help=_('Health monitor to disassociate.'))
        parser.add_argument('pool_id', metavar='POOL', help=_('ID of the pool to be disassociated with the health monitor.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        pool_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'pool', parsed_args.pool_id)
        neutron_client.disassociate_health_monitor(pool_id, parsed_args.health_monitor_id)
        print(_('Disassociated health monitor %s') % parsed_args.health_monitor_id, file=self.app.stdout)