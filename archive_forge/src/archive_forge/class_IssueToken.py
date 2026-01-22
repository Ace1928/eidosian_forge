from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
class IssueToken(command.ShowOne):
    _description = _('Issue new token')
    required_scope = False

    def get_parser(self, prog_name):
        parser = super(IssueToken, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        auth_ref = self.app.client_manager.auth_ref
        if not auth_ref:
            raise exceptions.AuthorizationFailure('Only an authorized user may issue a new token.')
        data = {}
        if auth_ref.auth_token:
            data['id'] = auth_ref.auth_token
        if auth_ref.expires:
            datetime_obj = auth_ref.expires
            expires_str = datetime_obj.strftime('%Y-%m-%dT%H:%M:%S%z')
            data['expires'] = expires_str
        if auth_ref.project_id:
            data['project_id'] = auth_ref.project_id
        if auth_ref.user_id:
            data['user_id'] = auth_ref.user_id
        return zip(*sorted(data.items()))