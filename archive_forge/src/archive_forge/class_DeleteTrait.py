from osc_lib.command import command
from osc_placement import version
class DeleteTrait(command.Command):
    """Delete the trait specified by {name}.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(DeleteTrait, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the trait.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = '/'.join([BASE_URL, parsed_args.name])
        http.request('DELETE', url)