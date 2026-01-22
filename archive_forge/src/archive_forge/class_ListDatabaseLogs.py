from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient import exceptions
from troveclient.i18n import _
class ListDatabaseLogs(command.Lister):
    _description = _('Lists the log files available for instance.')
    columns = ['Name', 'Type', 'Status', 'Published', 'Pending', 'Container', 'Prefix']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseLogs, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        database_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(database_instances, parsed_args.instance)
        log_list = database_instances.log_list(instance)
        logs = [osc_utils.get_item_properties(l, self.columns) for l in log_list]
        return (self.columns, logs)