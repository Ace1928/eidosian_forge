import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListVolumeGroupSnapshot(command.Lister):
    """Lists all volume group snapshot.

    This command requires ``--os-volume-api-version`` 3.14 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--all-projects', dest='all_projects', action='store_true', default=utils.env('ALL_PROJECTS', default=False), help=_('Shows details for all projects (admin only).'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.14'):
            msg = _("--os-volume-api-version 3.14 or greater is required to support the 'volume group snapshot list' command")
            raise exceptions.CommandError(msg)
        groups = volume_client.group_snapshots(all_projects=parsed_args.all_projects)
        column_headers = ('ID', 'Status', 'Name')
        columns = ('id', 'status', 'name')
        return (column_headers, (utils.get_item_properties(a, columns) for a in groups))