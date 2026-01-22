from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
class GetQuotaClass(command.ShowOne):
    """List the quotas for a quota class"""
    log = logging.getLogger(__name__ + '.GetQuotaClass')

    def get_parser(self, prog_name):
        parser = super(GetQuotaClass, self).get_parser(prog_name)
        parser.add_argument('quota_class_name', metavar='<quota_class_name>', help='The name of quota class')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        quota_class_name = parsed_args.quota_class_name
        quota_class = client.quota_classes.get(quota_class_name)
        columns = _quota_class_columns(quota_class)
        return (columns, utils.get_item_properties(quota_class, columns))