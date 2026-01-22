from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
class DeleteQuotas(command.Command):
    _description = _('Delete specified resource quota.')

    def get_parser(self, prog_name):
        parser = super(DeleteQuotas, self).get_parser(prog_name)
        parser.add_argument('--project-id', required=True, metavar='<project-id>', help='Project ID')
        parser.add_argument('--resource', required=True, metavar='<resource>', help='Resource name.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        try:
            mag_client.quotas.delete(parsed_args.project_id, parsed_args.resource)
            print('Request to delete quota for project id %(id)s and resource %(res)s has been accepted.' % {'id': parsed_args.project_id, 'res': parsed_args.resource})
        except Exception as e:
            print('Quota delete failed for project id %(id)s and resource %(res)s :%(e)s' % {'id': parsed_args.project_id, 'res': parsed_args.resource, 'e': e})