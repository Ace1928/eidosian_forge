import logging
from osc_lib.command import command
from osc_lib import exceptions
from neutronclient._i18n import _
class NetworkOnboardSubnets(command.Command):
    """Onboard network subnets into a subnet pool"""

    def get_parser(self, prog_name):
        parser = super(NetworkOnboardSubnets, self).get_parser(prog_name)
        parser.add_argument('network', metavar='<network>', help=_('Onboard all subnets associated with this network'))
        parser.add_argument('subnetpool', metavar='<subnetpool>', help=_('Target subnet pool for onboarding subnets'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.neutronclient
        subnetpool_id = _get_id(client, parsed_args.subnetpool, 'subnetpool')
        network_id = _get_id(client, parsed_args.network, 'network')
        body = {'network_id': network_id}
        try:
            client.onboard_network_subnets(subnetpool_id, body)
        except Exception as e:
            msg = _("Failed to onboard subnets for network '%(n)s': %(e)s") % {'n': parsed_args.network, 'e': e}
            raise exceptions.CommandError(msg)