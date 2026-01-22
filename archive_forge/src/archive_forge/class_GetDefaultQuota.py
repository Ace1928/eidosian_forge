from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.i18n import _
class GetDefaultQuota(command.ShowOne):
    """Get default quota of the project"""
    log = logging.getLogger(__name__ + '.GetDefaultQuota')

    def get_parser(self, prog_name):
        parser = super(GetDefaultQuota, self).get_parser(prog_name)
        parser.add_argument('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        default_quota = client.quotas.defaults(parsed_args.project_id)
        columns = _quota_columns(default_quota)
        return (columns, utils.get_item_properties(default_quota, columns))