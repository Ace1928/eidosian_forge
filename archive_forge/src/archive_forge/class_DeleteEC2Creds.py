import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteEC2Creds(command.Command):
    _description = _('Delete EC2 credentials')

    def get_parser(self, prog_name):
        parser = super(DeleteEC2Creds, self).get_parser(prog_name)
        parser.add_argument('access_keys', metavar='<access-key>', nargs='+', help=_('Credentials access key(s)'))
        parser.add_argument('--user', metavar='<user>', help=_('Delete credentials for user (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.user:
            user = utils.find_resource(identity_client.users, parsed_args.user).id
        else:
            user = self.app.client_manager.auth_ref.user_id
        result = 0
        for access_key in parsed_args.access_keys:
            try:
                identity_client.ec2.delete(user, access_key)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete EC2 credentials with access key '%(access_key)s': %(e)s"), {'access_key': access_key, 'e': e})
        if result > 0:
            total = len(parsed_args.access_keys)
            msg = _('%(result)s of %(total)s EC2 keys failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)