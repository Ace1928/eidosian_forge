import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class ShowLimit(command.ShowOne):
    _description = _('Display limit details')

    def get_parser(self, prog_name):
        parser = super(ShowLimit, self).get_parser(prog_name)
        parser.add_argument('limit_id', metavar='<limit-id>', help=_('Limit to display (ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        limit = identity_client.limits.get(parsed_args.limit_id)
        limit._info.pop('links', None)
        return zip(*sorted(limit._info.items()))