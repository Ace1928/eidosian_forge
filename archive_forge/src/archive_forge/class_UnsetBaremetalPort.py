import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class UnsetBaremetalPort(command.Command):
    """Unset baremetal port properties."""
    log = logging.getLogger(__name__ + '.UnsetBaremetalPort')

    def get_parser(self, prog_name):
        parser = super(UnsetBaremetalPort, self).get_parser(prog_name)
        parser.add_argument('port', metavar='<port>', help=_('UUID of the port.'))
        parser.add_argument('--extra', metavar='<key>', action='append', help=_('Extra to unset on this baremetal port (repeat option to unset multiple extras)'))
        parser.add_argument('--port-group', action='store_true', dest='portgroup', help=_('Remove port from the port group'))
        parser.add_argument('--physical-network', action='store_true', dest='physical_network', help=_('Unset the physical network on this baremetal port.'))
        parser.add_argument('--is-smartnic', dest='is_smartnic', action='store_true', help=_('Set Port as not Smart NIC port'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('remove', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.portgroup:
            properties.extend(utils.args_array_to_patch('remove', ['portgroup_uuid']))
        if parsed_args.physical_network:
            properties.extend(utils.args_array_to_patch('remove', ['physical_network']))
        if parsed_args.is_smartnic:
            properties.extend(utils.args_array_to_patch('add', ['is_smartnic=False']))
        if properties:
            baremetal_client.port.update(parsed_args.port, properties)
        else:
            self.log.warning('Please specify what to unset.')