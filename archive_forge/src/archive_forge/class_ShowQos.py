import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowQos(command.ShowOne):
    _description = _('Display QoS specification details')

    def get_parser(self, prog_name):
        parser = super(ShowQos, self).get_parser(prog_name)
        parser.add_argument('qos_spec', metavar='<qos-spec>', help=_('QoS specification to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_spec = utils.find_resource(volume_client.qos_specs, parsed_args.qos_spec)
        qos_associations = volume_client.qos_specs.get_associations(qos_spec)
        if qos_associations:
            associations = [association.name for association in qos_associations]
            qos_spec._info.update({'associations': format_columns.ListColumn(associations)})
        qos_spec._info.update({'properties': format_columns.DictColumn(qos_spec._info.pop('specs'))})
        return zip(*sorted(qos_spec._info.items()))