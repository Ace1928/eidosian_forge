import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalChassis(command.Command):
    """Set chassis properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalChassis')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalChassis, self).get_parser(prog_name)
        parser.add_argument('chassis', metavar='<chassis>', help=_('UUID of the chassis'))
        parser.add_argument('--description', metavar='<description>', help=_('Set the description of the chassis'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra to set on this chassis (repeat option to set multiple extras)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.description:
            description = ['description=%s' % parsed_args.description]
            properties.extend(utils.args_array_to_patch('add', description))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if properties:
            baremetal_client.chassis.update(parsed_args.chassis, properties)
        else:
            self.log.warning('Please specify what to set.')