import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class AbandonShareSnapshot(command.Command):
    """Abandon one or more share snapshots (Admin only)."""
    _description = _('Abandon share snapshot(s)')

    def get_parser(self, prog_name):
        parser = super(AbandonShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', nargs='+', help=_('Name or ID of the snapshot(s) to be abandoned.'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until share snapshot is abandoned'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for snapshot in parsed_args.snapshot:
            snapshot_obj = utils.find_resource(share_client.share_snapshots, snapshot)
            try:
                share_client.share_snapshots.unmanage(snapshot_obj)
                if parsed_args.wait:
                    if not utils.wait_for_delete(manager=share_client.share_snapshots, res_id=snapshot_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to abandon share snapshot with name or ID '%(snapshot)s': %(e)s"), {'snapshot': snapshot, 'e': e})
        if result > 0:
            total = len(parsed_args.snapshot)
            msg = _('%(result)s of %(total)s snapshots failed to abandon.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)