import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateCredential(command.ShowOne):
    _description = _('Create new credential')

    def get_parser(self, prog_name):
        parser = super(CreateCredential, self).get_parser(prog_name)
        parser.add_argument('user', metavar='<user>', help=_('user that owns the credential (name or ID)'))
        parser.add_argument('--type', default='cert', metavar='<type>', help=_('New credential type: cert, ec2, totp and so on'))
        parser.add_argument('data', metavar='<data>', help=_('New credential data'))
        parser.add_argument('--project', metavar='<project>', help=_('Project which limits the scope of the credential (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        user_id = utils.find_resource(identity_client.users, parsed_args.user).id
        if parsed_args.project:
            project = utils.find_resource(identity_client.projects, parsed_args.project).id
        else:
            project = None
        credential = identity_client.credentials.create(user=user_id, type=parsed_args.type, blob=parsed_args.data, project=project)
        credential._info.pop('links')
        return zip(*sorted(credential._info.items()))