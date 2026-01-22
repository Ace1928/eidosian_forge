import ast
import logging
from cliff import command
from cliff.formatters import table
from cliff import lister
from cliff import show
from blazarclient import exception
from blazarclient import utils
class UpdatePropertyCommand(BlazarCommand):
    api = 'reservation'
    resource = None
    log = None

    def run(self, parsed_args):
        self.log.debug('run(%s)' % parsed_args)
        blazar_client = self.get_client()
        body = self.args2body(parsed_args)
        resource_manager = getattr(blazar_client, self.resource)
        resource_manager.set_property(**body)
        print('Updated %s property: %s' % (self.resource, parsed_args.property_name), file=self.app.stdout)
        return

    def get_parser(self, prog_name):
        parser = super(UpdatePropertyCommand, self).get_parser(prog_name)
        parser.add_argument('property_name', metavar='PROPERTY_NAME', help='Name of property to patch.')
        parser.add_argument('--private', action='store_true', default=False, help='Set property to private.')
        parser.add_argument('--public', action='store_true', default=False, help='Set property to public.')
        return parser

    def args2body(self, parsed_args):
        return dict(property_name=parsed_args.property_name, private=parsed_args.private is True)