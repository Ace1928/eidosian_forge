import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ShowServerGroup(command.ShowOne):
    _description = _('Display server group details.')

    def get_parser(self, prog_name):
        parser = super(ShowServerGroup, self).get_parser(prog_name)
        parser.add_argument('server_group', metavar='<server-group>', help=_('server group to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        group = compute_client.find_server_group(parsed_args.server_group, ignore_missing=False)
        display_columns, columns = _get_server_group_columns(group, compute_client)
        data = utils.get_item_properties(group, columns, formatters=_formatters)
        return (display_columns, data)