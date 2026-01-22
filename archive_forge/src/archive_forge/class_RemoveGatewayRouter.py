import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class RemoveGatewayRouter(neutronV20.NeutronCommand):
    """Remove an external network gateway from a router."""
    resource = 'router'

    def get_parser(self, prog_name):
        parser = super(RemoveGatewayRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='ROUTER', help=_('ID or name of the router.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _router_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.router)
        neutron_client.remove_gateway_router(_router_id)
        print(_('Removed gateway from router %s') % parsed_args.router, file=self.app.stdout)