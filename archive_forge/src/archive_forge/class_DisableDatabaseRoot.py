from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class DisableDatabaseRoot(command.Command):
    _description = _('Disables root for an instance.')

    def get_parser(self, prog_name):
        parser = super(DisableDatabaseRoot, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        database_client_manager = self.app.client_manager.database
        db_instances = database_client_manager.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        db_root = database_client_manager.root
        db_root.disable_instance_root(instance)