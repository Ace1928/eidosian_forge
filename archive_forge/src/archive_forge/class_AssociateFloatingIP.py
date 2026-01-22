import argparse
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
class AssociateFloatingIP(neutronV20.NeutronCommand):
    """Create a mapping between a floating IP and a fixed IP."""
    resource = 'floatingip'

    def get_parser(self, prog_name):
        parser = super(AssociateFloatingIP, self).get_parser(prog_name)
        parser.add_argument('floatingip_id', metavar='FLOATINGIP_ID', help=_('ID of the floating IP to associate.'))
        parser.add_argument('port_id', metavar='PORT', help=_('ID or name of the port to be associated with the floating IP.'))
        parser.add_argument('--fixed-ip-address', help=_('IP address on the port (only required if port has multiple IPs).'))
        parser.add_argument('--fixed_ip_address', help=argparse.SUPPRESS)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        update_dict = {}
        neutronV20.update_dict(parsed_args, update_dict, ['port_id', 'fixed_ip_address'])
        neutron_client.update_floatingip(parsed_args.floatingip_id, {'floatingip': update_dict})
        print(_('Associated floating IP %s') % parsed_args.floatingip_id, file=self.app.stdout)