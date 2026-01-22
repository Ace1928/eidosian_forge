import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalChassis(command.ShowOne):
    """Create a new chassis."""
    log = logging.getLogger(__name__ + '.CreateBaremetalChassis')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalChassis, self).get_parser(prog_name)
        parser.add_argument('--description', dest='description', metavar='<description>', help=_('Description for the chassis'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        parser.add_argument('--uuid', metavar='<uuid>', help=_('Unique UUID of the chassis'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        field_list = ['description', 'extra', 'uuid']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and (not v is None)))
        fields = utils.args_array_to_dict(fields, 'extra')
        chassis = baremetal_client.chassis.create(**fields)._info
        chassis.pop('links', None)
        chassis.pop('nodes', None)
        return self.dict2columns(chassis)