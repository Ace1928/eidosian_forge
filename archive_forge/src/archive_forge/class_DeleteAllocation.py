from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_placement import version
class DeleteAllocation(command.Command):
    """Delete all resource allocations for a given consumer."""

    def get_parser(self, prog_name):
        parser = super(DeleteAllocation, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the consumer')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        http.request('DELETE', url)