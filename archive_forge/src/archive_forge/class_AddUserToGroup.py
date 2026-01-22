import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class AddUserToGroup(command.Command):
    _description = _('Add user to group')

    def get_parser(self, prog_name):
        parser = super(AddUserToGroup, self).get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Group to contain <user> (name or ID)'))
        parser.add_argument('user', metavar='<user>', nargs='+', help=_('User(s) to add to <group> (name or ID) (repeat option to add multiple users)'))
        common.add_group_domain_option_to_parser(parser)
        common.add_user_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        group_id = common.find_group(identity_client, parsed_args.group, parsed_args.group_domain).id
        result = 0
        for i in parsed_args.user:
            try:
                user_id = common.find_user(identity_client, i, parsed_args.user_domain).id
                identity_client.users.add_to_group(user_id, group_id)
            except Exception as e:
                result += 1
                msg = _('%(user)s not added to group %(group)s: %(e)s') % {'user': i, 'group': parsed_args.group, 'e': e}
                LOG.error(msg)
        if result > 0:
            total = len(parsed_args.user)
            msg = _('%(result)s of %(total)s users not added to group %(group)s.') % {'result': result, 'total': total, 'group': parsed_args.group}
            raise exceptions.CommandError(msg)