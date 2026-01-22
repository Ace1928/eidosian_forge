from osc_lib.command import command
from osc_placement import version
class CreateTrait(command.Command):
    """Create a new custom trait.

    Custom traits must begin with the prefix ``CUSTOM_`` and contain only the
    letters A through Z, the numbers 0 through 9 and the underscore "_"
    character.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(CreateTrait, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the trait.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = '/'.join([BASE_URL, parsed_args.name])
        http.request('PUT', url)