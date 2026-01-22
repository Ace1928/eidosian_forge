import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
class SetBgpvpn(command.Command):
    _description = _('Set BGP VPN properties')

    def get_parser(self, prog_name):
        parser = super(SetBgpvpn, self).get_parser(prog_name)
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN to update (name or ID)'))
        _get_common_parser(parser, update='set')
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        id = client.find_bgpvpn(parsed_args.bgpvpn)['id']
        body = _args2body(self.app.client_manager, id, 'set', parsed_args)
        client.update_bgpvpn(id, **body)