import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class DeleteServerGroup(command.Command):
    _description = _('Delete existing server group(s).')

    def get_parser(self, prog_name):
        parser = super(DeleteServerGroup, self).get_parser(prog_name)
        parser.add_argument('server_group', metavar='<server-group>', nargs='+', help=_('server group(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        result = 0
        for group in parsed_args.server_group:
            try:
                group_obj = compute_client.find_server_group(group, ignore_missing=False)
                compute_client.delete_server_group(group_obj.id)
            except Exception as e:
                result += 1
                LOG.error(e)
        if result > 0:
            total = len(parsed_args.server_group)
            msg = _('%(result)s of %(total)s server groups failed to delete.')
            raise exceptions.CommandError(msg % {'result': result, 'total': total})