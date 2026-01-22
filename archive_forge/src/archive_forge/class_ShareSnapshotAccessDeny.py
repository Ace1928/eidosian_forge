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
class ShareSnapshotAccessDeny(command.Command):
    """Delete access to a snapshot"""
    _description = _('Delete access to a snapshot')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotAccessDeny, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the share snapshot to deny access to.'))
        parser.add_argument('id', metavar='<id>', nargs='+', help=_('ID(s) of the access rule(s) to be deleted.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        snapshot_obj = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        for access_id in parsed_args.id:
            try:
                share_client.share_snapshots.deny(snapshot=snapshot_obj, id=access_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete access to share snapshot for an access rule with ID '%(access)s': %(e)s"), {'access': access_id, 'e': e})
        if result > 0:
            total = len(parsed_args.id)
            msg = _("Failed to delete access to a share snapshot for %(result)s out of %(total)s access rule ID's ") % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)