import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ExportBackupRecord(command.ShowOne):
    _description = _('Export volume backup details.\n\nBackup information can be imported into a new service instance to be able to\nrestore.')

    def get_parser(self, prog_name):
        parser = super(ExportBackupRecord, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Backup to export (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        backup = utils.find_resource(volume_client.backups, parsed_args.backup)
        backup_data = volume_client.backups.export_record(backup.id)
        if parsed_args.formatter == 'table':
            backup_data['Backup Service'] = backup_data.pop('backup_service')
            backup_data['Metadata'] = backup_data.pop('backup_url')
        return zip(*sorted(backup_data.items()))