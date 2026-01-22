import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class ListBaremetalDriverProperty(command.Lister):
    """List the driver properties."""
    log = logging.getLogger(__name__ + '.ListBaremetalDriverProperty')

    def get_parser(self, prog_name):
        parser = super(ListBaremetalDriverProperty, self).get_parser(prog_name)
        parser.add_argument('driver', metavar='<driver>', help='Name of the driver.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        driver_properties = baremetal_client.driver.properties(parsed_args.driver)
        labels = ['Property', 'Description']
        return (labels, sorted(driver_properties.items()))