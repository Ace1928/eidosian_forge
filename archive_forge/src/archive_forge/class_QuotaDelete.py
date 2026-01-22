from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class QuotaDelete(command.Command):
    """Delete quota for project/user or project/share-type.

    The quota will revert back to default.
    """
    _description = _('Delete Quota')

    def get_parser(self, prog_name):
        parser = super(QuotaDelete, self).get_parser(prog_name)
        quota_type = parser.add_mutually_exclusive_group()
        parser.add_argument('project', metavar='<project>', help=_('Name or ID of the project to delete quotas for.'))
        quota_type.add_argument('--user', metavar='<user>', default=None, help=_("Name or ID of user to delete the quotas for. Optional. Mutually exclusive with '--share-type'."))
        quota_type.add_argument('--share-type', metavar='<share-type>', type=str, default=None, help=_("Name or ID of a share type to delete the quotas for. Optional. Mutually exclusive with '--user'. Available only for microversion >= 2.39"))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        user_id = None
        if parsed_args.user:
            user_id = utils.find_resource(identity_client.users, parsed_args.user).id
        project_id = utils.find_resource(identity_client.projects, parsed_args.project).id
        kwargs = {'tenant_id': project_id, 'user_id': user_id}
        if parsed_args.share_type:
            if share_client.api_version < api_versions.APIVersion('2.39'):
                raise exceptions.CommandError(_("'share type' quotas are available only starting with API microversion '2.39'."))
            kwargs['share_type'] = parsed_args.share_type
        share_client.quotas.delete(**kwargs)