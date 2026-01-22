import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CheckUserInGroup(command.Command):
    _description = _('Check user membership in group')

    def get_parser(self, prog_name):
        parser = super(CheckUserInGroup, self).get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Group to check (name or ID)'))
        parser.add_argument('user', metavar='<user>', help=_('User to check (name or ID)'))
        common.add_group_domain_option_to_parser(parser)
        common.add_user_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        user_id = common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        group_id = common.find_group(identity_client, parsed_args.group, parsed_args.group_domain).id
        try:
            identity_client.users.check_in_group(user_id, group_id)
        except ks_exc.http.HTTPClientError as e:
            if e.http_status == 403 or e.http_status == 404:
                msg = _('%(user)s not in group %(group)s\n') % {'user': parsed_args.user, 'group': parsed_args.group}
                self.app.stderr.write(msg)
            else:
                raise e
        else:
            msg = _('%(user)s in group %(group)s\n') % {'user': parsed_args.user, 'group': parsed_args.group}
            self.app.stdout.write(msg)