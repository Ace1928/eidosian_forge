import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class DeleteShareGroup(command.Command):
    """Delete one or more share groups."""
    _description = _('Delete one or more share groups')

    def get_parser(self, prog_name):
        parser = super(DeleteShareGroup, self).get_parser(prog_name)
        parser.add_argument('share_group', metavar='<share_group>', nargs='+', help=_('Name or ID of the share group(s) to delete'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Attempt to force delete the share group (Default=False) (Admin only).'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share group to delete'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for share_group in parsed_args.share_group:
            try:
                share_group_obj = osc_utils.find_resource(share_client.share_groups, share_group)
                share_client.share_groups.delete(share_group_obj, force=parsed_args.force)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_groups, res_id=share_group_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete share group with name or ID '%(share_group)s': %(e)s"), {'share_group': share_group, 'e': e})
        if result > 0:
            total = len(parsed_args.share_group)
            msg = _('%(result)s of %(total)s share groups failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)