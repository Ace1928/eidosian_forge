import argparse
from osc_lib.command import command
from osc_lib import utils
from osc_placement.resources import common
from osc_placement import version
class DeleteResourceProvider(command.Command):
    """Delete a resource provider"""

    def get_parser(self, prog_name):
        parser = super(DeleteResourceProvider, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        http.request('DELETE', url)