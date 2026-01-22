import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class RemoveAggregateHost(command.ShowOne):
    _description = _('Remove host from aggregate')

    def get_parser(self, prog_name):
        parser = super(RemoveAggregateHost, self).get_parser(prog_name)
        parser.add_argument('aggregate', metavar='<aggregate>', help=_('Aggregate (name or ID)'))
        parser.add_argument('host', metavar='<host>', help=_('Host to remove from <aggregate>'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        aggregate = compute_client.find_aggregate(parsed_args.aggregate, ignore_missing=False)
        aggregate = compute_client.remove_host_from_aggregate(aggregate.id, parsed_args.host)
        display_columns, columns = _get_aggregate_columns(aggregate)
        data = utils.get_item_properties(aggregate, columns, formatters=_aggregate_formatters)
        return (display_columns, data)