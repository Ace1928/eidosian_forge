from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class CreateResourceClass(command.Command):
    """Create a new resource class.

    This command requires at least ``--os-placement-api-version 1.2``.
    """

    def get_parser(self, prog_name):
        parser = super(CreateResourceClass, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the resource class')
        return parser

    @version.check(version.ge('1.2'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        http.request('POST', BASE_URL, json={'name': parsed_args.name})