import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateQos(command.ShowOne):
    _description = _('Create new QoS specification')

    def get_parser(self, prog_name):
        parser = super(CreateQos, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('New QoS specification name'))
        consumer_choices = ['front-end', 'back-end', 'both']
        parser.add_argument('--consumer', metavar='<consumer>', choices=consumer_choices, default='both', help=_("Consumer of the QoS. Valid consumers: %s (defaults to 'both')") % utils.format_list(consumer_choices))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Set a QoS specification property (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        specs = {}
        specs.update({'consumer': parsed_args.consumer})
        if parsed_args.property:
            specs.update(parsed_args.property)
        qos_spec = volume_client.qos_specs.create(parsed_args.name, specs)
        qos_spec._info.update({'properties': format_columns.DictColumn(qos_spec._info.pop('specs'))})
        return zip(*sorted(qos_spec._info.items()))