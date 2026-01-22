import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalPortGroup(command.Command):
    """Set baremetal port group properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalPortGroup')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalPortGroup, self).get_parser(prog_name)
        parser.add_argument('portgroup', metavar='<port group>', help=_('Name or UUID of the port group.'))
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', help=_('Update UUID of the node that this port group belongs to.'))
        parser.add_argument('--address', metavar='<mac-address>', help=_('MAC address for this port group.'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of the port group.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra to set on this baremetal port group (repeat option to set multiple extras).'))
        parser.add_argument('--mode', help=_('Mode of the port group. For possible values, refer to https://www.kernel.org/doc/Documentation/networking/bonding.txt.'))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action='append', help=_("Key/value property related to this port group's configuration (repeat option to set multiple properties)."))
        standalone_ports_group = parser.add_mutually_exclusive_group()
        standalone_ports_group.add_argument('--support-standalone-ports', action='store_true', default=None, help=_('Ports that are members of this port group can be used as stand-alone ports.'))
        standalone_ports_group.add_argument('--unsupport-standalone-ports', action='store_true', help=_('Ports that are members of this port group cannot be used as stand-alone ports.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.node_uuid:
            properties.extend(utils.args_array_to_patch('add', ['node_uuid=%s' % parsed_args.node_uuid]))
        if parsed_args.address:
            properties.extend(utils.args_array_to_patch('add', ['address=%s' % parsed_args.address]))
        if parsed_args.name:
            name = ['name=%s' % parsed_args.name]
            properties.extend(utils.args_array_to_patch('add', name))
        if parsed_args.support_standalone_ports:
            properties.extend(utils.args_array_to_patch('add', ['standalone_ports_supported=True']))
        if parsed_args.unsupport_standalone_ports:
            properties.extend(utils.args_array_to_patch('add', ['standalone_ports_supported=False']))
        if parsed_args.mode:
            properties.extend(utils.args_array_to_patch('add', ['mode="%s"' % parsed_args.mode]))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.properties:
            properties.extend(utils.args_array_to_patch('add', ['properties/' + x for x in parsed_args.properties]))
        if properties:
            baremetal_client.portgroup.update(parsed_args.portgroup, properties)
        else:
            self.log.warning('Please specify what to set.')