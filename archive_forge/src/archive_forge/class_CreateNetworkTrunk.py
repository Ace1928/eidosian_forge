import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class CreateNetworkTrunk(command.ShowOne):
    """Create a network trunk for a given project"""

    def get_parser(self, prog_name):
        parser = super(CreateNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the trunk to create'))
        parser.add_argument('--description', metavar='<description>', help=_('A description of the trunk'))
        parser.add_argument('--parent-port', metavar='<parent-port>', required=True, help=_('Parent port belonging to this trunk (name or ID)'))
        parser.add_argument('--subport', metavar='<port=,segmentation-type=,segmentation-id=>', action=parseractions.MultiKeyValueAction, dest='add_subports', optional_keys=['segmentation-id', 'segmentation-type'], required_keys=['port'], help=_("Subport to add. Subport is of form 'port=<name or ID>,segmentation-type=<segmentation-type>,segmentation-id=<segmentation-ID>' (--subport) option can be repeated"))
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=True, help=_('Enable trunk (default)'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable trunk'))
        identity_utils.add_project_owner_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs_for_trunk(self.app.client_manager, parsed_args)
        obj = client.create_trunk(**attrs)
        display_columns, columns = _get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)