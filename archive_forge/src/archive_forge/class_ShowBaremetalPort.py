import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalPort(command.ShowOne):
    """Show baremetal port details."""
    log = logging.getLogger(__name__ + '.ShowBaremetalPort')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalPort, self).get_parser(prog_name)
        parser.add_argument('port', metavar='<id>', help=_('UUID of the port (or MAC address if --address is specified).'))
        parser.add_argument('--address', dest='address', action='store_true', default=False, help=_('<id> is the MAC address (instead of the UUID) of the port.'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.PORT_DETAILED_RESOURCE.fields, default=[], help=_('One or more port fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        if parsed_args.address:
            port = baremetal_client.port.get_by_address(parsed_args.port, fields=fields)._info
        else:
            port = baremetal_client.port.get(parsed_args.port, fields=fields)._info
        port.pop('links', None)
        return zip(*sorted(port.items()))