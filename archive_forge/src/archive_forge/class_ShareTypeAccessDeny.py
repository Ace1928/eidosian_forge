from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShareTypeAccessDeny(command.Command):
    """Delete access from share type."""
    _description = _('Delete access from share type')

    def get_parser(self, prog_name):
        parser = super(ShareTypeAccessDeny, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Share type name or ID to delete access from'))
        parser.add_argument('project_id', metavar='<project_id>', help=_('Project ID to delete share type access for'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        try:
            share_client.share_type_access.remove_project_access(share_type, parsed_args.project_id)
        except Exception as e:
            raise exceptions.CommandError('Failed to remove access from share type : %s' % e)