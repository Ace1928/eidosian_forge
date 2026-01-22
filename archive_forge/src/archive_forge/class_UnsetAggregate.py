import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class UnsetAggregate(command.Command):
    _description = _('Unset aggregate properties')

    def get_parser(self, prog_name):
        parser = super(UnsetAggregate, self).get_parser(prog_name)
        parser.add_argument('aggregate', metavar='<aggregate>', help=_('Aggregate to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', default=[], dest='properties', help=_('Property to remove from aggregate (repeat option to remove multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        aggregate = compute_client.find_aggregate(parsed_args.aggregate, ignore_missing=False)
        properties = {key: None for key in parsed_args.properties}
        if properties:
            compute_client.set_aggregate_metadata(aggregate.id, properties)