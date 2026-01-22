import copy
import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from openstack import utils as sdk_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ShowVolumeBackup(command.ShowOne):
    _description = _('Display volume backup details')

    def get_parser(self, prog_name):
        parser = super(ShowVolumeBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Backup to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        backup = volume_client.get_backup(parsed_args.backup)
        columns = ('availability_zone', 'container', 'created_at', 'data_timestamp', 'description', 'encryption_key_id', 'fail_reason', 'has_dependent_backups', 'id', 'is_incremental', 'metadata', 'name', 'object_count', 'project_id', 'size', 'snapshot_id', 'status', 'updated_at', 'user_id', 'volume_id')
        data = utils.get_dict_properties(backup, columns)
        return (columns, data)