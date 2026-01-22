import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class ShowBaremetalDriver(command.ShowOne):
    """Show information about a driver."""
    log = logging.getLogger(__name__ + '.ShowBaremetalDriver')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalDriver, self).get_parser(prog_name)
        parser.add_argument('driver', metavar='<driver>', help=_('Name of the driver.'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.DRIVER_DETAILED_RESOURCE.fields, help=_('One or more node fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        driver = baremetal_client.driver.get(parsed_args.driver, fields=fields)._info
        driver.pop('links', None)
        driver.pop('properties', None)
        driver = utils.convert_list_props_to_comma_separated(driver)
        return zip(*sorted(driver.items()))