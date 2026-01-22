from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
class UpdateQuotas(command.Command):
    _description = _('Update information about the given project resource quota.')

    def get_parser(self, prog_name):
        parser = super(UpdateQuotas, self).get_parser(prog_name)
        parser.add_argument('--project-id', required=True, metavar='<project-id>', help='Project ID')
        parser.add_argument('--resource', required=True, metavar='<resource>', help='Resource name.')
        parser.add_argument('--hard-limit', metavar='<hard-limit>', type=int, default=1, help='Max resource limit (default: hard-limit=1)')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        opts = {'project_id': parsed_args.project_id, 'resource': parsed_args.resource, 'hard_limit': parsed_args.hard_limit}
        try:
            quota = mag_client.quotas.update(parsed_args.project_id, parsed_args.resource, opts)
            _show_quota(quota)
        except Exception as e:
            print('Update quota for project_id %(id)s resource %(res)s failed: %(e)s' % {'id': parsed_args.project_id, 'res': parsed_args.resource, 'e': e})