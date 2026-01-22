import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class SetGatewayRouter(neutronV20.NeutronCommand):
    """Set the external network gateway for a router."""
    resource = 'router'

    def get_parser(self, prog_name):
        parser = super(SetGatewayRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='ROUTER', help=_('ID or name of the router.'))
        parser.add_argument('external_network', metavar='EXTERNAL-NETWORK', help=_('ID or name of the external network for the gateway.'))
        parser.add_argument('--enable-snat', action='store_true', help=_('Enable source NAT on the router gateway.'))
        parser.add_argument('--disable-snat', action='store_true', help=_('Disable source NAT on the router gateway.'))
        parser.add_argument('--fixed-ip', metavar='subnet_id=SUBNET,ip_address=IP_ADDR', action='append', type=utils.str2dict_type(optional_keys=['subnet_id', 'ip_address']), help=_('Desired IP and/or subnet on external network: subnet_id=<name_or_id>,ip_address=<ip>. You can specify both of subnet_id and ip_address or specify one of them as well. You can repeat this option.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _router_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.router)
        _ext_net_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'network', parsed_args.external_network)
        router_dict = {'network_id': _ext_net_id}
        if parsed_args.enable_snat:
            router_dict['enable_snat'] = True
        if parsed_args.disable_snat:
            router_dict['enable_snat'] = False
        if parsed_args.fixed_ip:
            ips = []
            for ip_spec in parsed_args.fixed_ip:
                subnet_name_id = ip_spec.get('subnet_id')
                if subnet_name_id:
                    subnet_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'subnet', subnet_name_id)
                    ip_spec['subnet_id'] = subnet_id
                ips.append(ip_spec)
            router_dict['external_fixed_ips'] = ips
        neutron_client.add_gateway_router(_router_id, router_dict)
        print(_('Set gateway for router %s') % parsed_args.router, file=self.app.stdout)