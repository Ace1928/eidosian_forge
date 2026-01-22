import logging
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteTransferRequest(command.Command):
    _description = _('Delete volume transfer request(s).')

    def get_parser(self, prog_name):
        parser = super(DeleteTransferRequest, self).get_parser(prog_name)
        parser.add_argument('transfer_request', metavar='<transfer-request>', nargs='+', help=_('Volume transfer request(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for t in parsed_args.transfer_request:
            try:
                transfer_request_id = utils.find_resource(volume_client.transfers, t).id
                volume_client.transfers.delete(transfer_request_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete volume transfer request with name or ID '%(transfer)s': %(e)s") % {'transfer': t, 'e': e})
        if result > 0:
            total = len(parsed_args.transfer_request)
            msg = _('%(result)s of %(total)s volume transfer requests failed to delete') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)