from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
class ShowDatabaseFlavor(command.ShowOne):
    _description = _('Shows details of a database flavor')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', help=_('ID or name of the flavor'))
        return parser

    def take_action(self, parsed_args):
        db_flavors = self.app.client_manager.database.flavors
        flavor = utils.find_resource(db_flavors, parsed_args.flavor)
        flavor = set_attributes_for_print_detail(flavor)
        return zip(*sorted(list(flavor.items())))