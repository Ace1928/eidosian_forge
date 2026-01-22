import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetQos(command.Command):
    _description = _('Set QoS specification properties')

    def get_parser(self, prog_name):
        parser = super(SetQos, self).get_parser(prog_name)
        parser.add_argument('qos_spec', metavar='<qos-spec>', help=_('QoS specification to modify (name or ID)'))
        parser.add_argument('--no-property', dest='no_property', action='store_true', help=_('Remove all properties from <qos-spec> (specify both --no-property and --property to remove the current properties before setting new properties)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Property to add or modify for this QoS specification (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_spec = utils.find_resource(volume_client.qos_specs, parsed_args.qos_spec)
        result = 0
        if parsed_args.no_property:
            try:
                key_list = list(qos_spec._info['specs'].keys())
                volume_client.qos_specs.unset_keys(qos_spec.id, key_list)
            except Exception as e:
                LOG.error(_('Failed to clean qos properties: %s'), e)
                result += 1
        if parsed_args.property:
            try:
                volume_client.qos_specs.set_keys(qos_spec.id, parsed_args.property)
            except Exception as e:
                LOG.error(_('Failed to set qos property: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))