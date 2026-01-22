import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListVolumeGroupType(command.Lister):
    """Lists all volume group types.

    This command requires ``--os-volume-api-version`` 3.11 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--default', action='store_true', dest='show_default', default=False, help=_('List the default volume group type.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.11'):
            msg = _("--os-volume-api-version 3.11 or greater is required to support the 'volume group type list' command")
            raise exceptions.CommandError(msg)
        if parsed_args.show_default:
            group_types = [volume_client.group_types.default()]
        else:
            group_types = volume_client.group_types.list()
        column_headers = ('ID', 'Name', 'Is Public', 'Properties')
        columns = ('id', 'name', 'is_public', 'group_specs')
        return (column_headers, (utils.get_item_properties(a, columns) for a in group_types))