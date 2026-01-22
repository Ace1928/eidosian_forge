import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class UnsetNetworkTrunk(command.Command):
    """Unset subports from a given network trunk"""

    def get_parser(self, prog_name):
        parser = super(UnsetNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('trunk', metavar='<trunk>', help=_('Unset subports from this trunk (name or ID)'))
        parser.add_argument('--subport', metavar='<subport>', required=True, action='append', dest='unset_subports', help=_('Subport to delete (name or ID of the port) (--subport) option can be repeated'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs_for_subports(self.app.client_manager, parsed_args)
        trunk_id = client.find_trunk(parsed_args.trunk)
        client.delete_trunk_subports(trunk_id, attrs)