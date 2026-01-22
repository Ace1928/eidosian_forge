import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class DeleteShareTransfer(command.Command):
    """Remove one or more transfers."""
    _description = _('Remove one or more transfers')

    def get_parser(self, prog_name):
        parser = super(DeleteShareTransfer, self).get_parser(prog_name)
        parser.add_argument('transfer', metavar='<transfer>', nargs='+', help='Name(s) or ID(s) of the transfer(s).')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        failure_count = 0
        for transfer in parsed_args.transfer:
            try:
                transfer_obj = apiutils.find_resource(share_client.transfers, transfer)
                share_client.transfers.delete(transfer_obj.id)
            except Exception as e:
                failure_count += 1
                LOG.error(_('Failed to delete %(transfer)s: %(e)s'), {'transfer': transfer, 'e': e})
        if failure_count > 0:
            raise exceptions.CommandError(_('Unable to delete some or all of the specified transfers.'))