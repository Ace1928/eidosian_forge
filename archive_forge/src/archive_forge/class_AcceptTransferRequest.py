import logging
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class AcceptTransferRequest(command.ShowOne):
    _description = _('Accept volume transfer request.')

    def get_parser(self, prog_name):
        parser = super(AcceptTransferRequest, self).get_parser(prog_name)
        parser.add_argument('transfer_request', metavar='<transfer-request-id>', help=_('Volume transfer request to accept (ID only)'))
        parser.add_argument('--auth-key', metavar='<key>', required=True, help=_('Volume transfer request authentication key'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        try:
            transfer_request_id = utils.find_resource(volume_client.transfers, parsed_args.transfer_request).id
        except exceptions.CommandError:
            transfer_request_id = parsed_args.transfer_request
        transfer_accept = volume_client.transfers.accept(transfer_request_id, parsed_args.auth_key)
        transfer_accept._info.pop('links', None)
        return zip(*sorted(transfer_accept._info.items()))