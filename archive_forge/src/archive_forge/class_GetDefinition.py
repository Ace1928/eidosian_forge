import argparse
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class GetDefinition(command.Command):
    """Show workbook definition."""

    def get_parser(self, prog_name):
        parser = super(GetDefinition, self).get_parser(prog_name)
        parser.add_argument('name', help='Workbook name')
        return parser

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        definition = mistral_client.workbooks.get(parsed_args.name).definition
        self.app.stdout.write(definition or '\n')