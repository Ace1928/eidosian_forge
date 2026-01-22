import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteGroup(command.Command):
    _description = _('Delete group(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteGroup, self).get_parser(prog_name)
        parser.add_argument('groups', metavar='<group>', nargs='+', help=_('Group(s) to delete (name or ID)'))
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain containing group(s) (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for group in parsed_args.groups:
            try:
                group_obj = common.find_group(identity_client, group, parsed_args.domain)
                identity_client.groups.delete(group_obj.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete group with name or ID '%(group)s': %(e)s"), {'group': group, 'e': e})
        if errors > 0:
            total = len(parsed_args.groups)
            msg = _('%(errors)s of %(total)s groups failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)