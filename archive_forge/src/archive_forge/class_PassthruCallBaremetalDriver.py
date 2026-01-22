import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class PassthruCallBaremetalDriver(command.ShowOne):
    """Call a vendor passthru method for a driver."""
    log = logging.getLogger(__name__ + '.PassthruCallBaremetalDriver')

    def get_parser(self, prog_name):
        parser = super(PassthruCallBaremetalDriver, self).get_parser(prog_name)
        parser.add_argument('driver', metavar='<driver>', help=_('Name of the driver.'))
        parser.add_argument('method', metavar='<method>', help=_('Vendor passthru method to be called.'))
        parser.add_argument('--arg', metavar='<key=value>', action='append', help=_('Argument to pass to the passthru method (repeat option to specify multiple arguments).'))
        parser.add_argument('--http-method', dest='http_method', metavar='<http-method>', choices=v1_utils.HTTP_METHODS, default='POST', help=_("The HTTP method to use in the passthru request. One of %s. Defaults to 'POST'.") % oscutils.format_list(v1_utils.HTTP_METHODS))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        arguments = utils.key_value_pairs_to_dict(parsed_args.arg)
        response = baremetal_client.driver.vendor_passthru(parsed_args.driver, parsed_args.method, http_method=parsed_args.http_method, args=arguments)
        return self.dict2columns(response)