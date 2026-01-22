import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.common import constants
class ShowShareServer(command.ShowOne):
    """Show share server (Admin only)."""
    _description = _('Show details about a share server (Admin only).')

    def get_parser(self, prog_name):
        parser = super(ShowShareServer, self).get_parser(prog_name)
        parser.add_argument('share_server', metavar='<share-server>', help=_('ID of share server.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_server = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server)
        if 'backend_details' in share_server._info:
            del share_server._info['backend_details']
        share_server._info.pop('links', None)
        return self.dict2columns(share_server._info)