from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class ListDatabaseBackupStrategies(command.Lister):
    _description = _('List backup strategies')
    columns = ['Project ID', 'Instance ID', 'Swift Container']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseBackupStrategies, self).get_parser(prog_name)
        parser.add_argument('--instance-id', help=_('Filter results by database instance ID.'))
        parser.add_argument('--project-id', help=_('Project ID in Keystone. Only admin user is allowed to list backup strategy for other projects.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database.backup_strategies
        result = manager.list(instance_id=parsed_args.instance_id, project_id=parsed_args.project_id)
        backup_strategies = [osc_utils.get_item_properties(item, self.columns) for item in result]
        return (self.columns, backup_strategies)