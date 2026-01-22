import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowMapping(command.ShowOne):
    _description = _('Display mapping details')

    def get_parser(self, prog_name):
        parser = super(ShowMapping, self).get_parser(prog_name)
        parser.add_argument('mapping', metavar='<mapping>', help=_('Mapping to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        mapping = identity_client.federation.mappings.get(parsed_args.mapping)
        mapping._info.pop('links', None)
        return zip(*sorted(mapping._info.items()))