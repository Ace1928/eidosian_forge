import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalAllocation(command.Command):
    """Set baremetal allocation properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalAllocation')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalAllocation, self).get_parser(prog_name)
        parser.add_argument('allocation', metavar='<allocation>', help=_('Name or UUID of the allocation'))
        parser.add_argument('--name', metavar='<name>', help=_('Set the name of the allocation'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra property to set on this allocation (repeat option to set multiple extra properties)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.name:
            properties.extend(utils.args_array_to_patch('add', ['name=%s' % parsed_args.name]))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if properties:
            baremetal_client.allocation.update(parsed_args.allocation, properties)
        else:
            self.log.warning('Please specify what to set.')