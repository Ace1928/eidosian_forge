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
class AbandonShareServer(command.Command):
    """Remove one or more share servers (Admin only)."""
    _description = _('Remove one or more share server(s) (Admin only).')

    def get_parser(self, prog_name):
        parser = super(AbandonShareServer, self).get_parser(prog_name)
        parser.add_argument('share_server', metavar='<share-server>', nargs='+', help=_('ID of the server(s) to be abandoned.'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Enforces the unmanage share server operation, even if the backend driver does not support it.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait until share server is abandoned'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for server in parsed_args.share_server:
            try:
                server_obj = osc_utils.find_resource(share_client.share_servers, server)
                kwargs = {}
                if parsed_args.force:
                    kwargs['force'] = parsed_args.force
                share_client.share_servers.unmanage(server_obj, **kwargs)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_servers, res_id=server_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to abandon share server with ID '%(server)s': %(e)s"), {'server': server, 'e': e})
        if result > 0:
            total = len(parsed_args.share_server)
            msg = f'Failed to abandon {result} of {total} servers.'
            raise exceptions.CommandError(_(msg))