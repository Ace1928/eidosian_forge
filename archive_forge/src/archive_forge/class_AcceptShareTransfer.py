import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class AcceptShareTransfer(command.Command):
    """Accepts a share transfer."""
    _description = _('Accepts a share transfer')

    def get_parser(self, prog_name):
        parser = super(AcceptShareTransfer, self).get_parser(prog_name)
        parser.add_argument('transfer', metavar='<transfer>', help='ID of transfer to accept.')
        parser.add_argument('auth_key', metavar='<auth_key>', help='Authentication key of transfer to accept.')
        parser.add_argument('--clear-rules', '--clear_rules', dest='clear_rules', action='store_true', default=False, help='Whether manila should clean up the access rules after the transfer is complete. (Default=False)')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_client.transfers.accept(parsed_args.transfer, parsed_args.auth_key, clear_access_rules=parsed_args.clear_rules)