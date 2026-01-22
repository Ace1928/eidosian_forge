import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class RouterInterfaceCommand(neutronV20.NeutronCommand):
    """Based class to Add/Remove router interface."""
    resource = 'router'

    def call_api(self, neutron_client, router_id, body):
        raise NotImplementedError()

    def success_message(self, router_id, portinfo):
        raise NotImplementedError()

    def get_parser(self, prog_name):
        parser = super(RouterInterfaceCommand, self).get_parser(prog_name)
        parser.add_argument('router', metavar='ROUTER', help=_('ID or name of the router.'))
        parser.add_argument('interface', metavar='INTERFACE', help=_('The format is "SUBNET|subnet=SUBNET|port=PORT". Either a subnet or port must be specified. Both ID and name are accepted as SUBNET or PORT. Note that "subnet=" can be omitted when specifying a subnet.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        if '=' in parsed_args.interface:
            resource, value = parsed_args.interface.split('=', 1)
            if resource not in ['subnet', 'port']:
                exceptions.CommandError(_('You must specify either subnet or port for INTERFACE parameter.'))
        else:
            resource = 'subnet'
            value = parsed_args.interface
        _router_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.router)
        _interface_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, resource, value)
        body = {'%s_id' % resource: _interface_id}
        portinfo = self.call_api(neutron_client, _router_id, body)
        print(self.success_message(parsed_args.router, portinfo), file=self.app.stdout)