from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient import exceptions
from troveclient.i18n import _
class ShowDatabaseInstanceLog(command.ShowOne):
    _description = _('Show information of given log name for the database instance.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseInstanceLog, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('Id or Name of the instance.'))
        parser.add_argument('log_name', metavar='<log_name>', type=str, help=_('Name of log to operate.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        log_info = db_instances.log_show(instance, parsed_args.log_name)
        result = log_info._info
        return zip(*sorted(result.items()))