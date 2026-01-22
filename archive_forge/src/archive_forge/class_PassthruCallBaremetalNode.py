import argparse
import itertools
import json
import logging
import sys
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class PassthruCallBaremetalNode(command.Command):
    """Call a vendor passthru method for a node"""
    log = logging.getLogger(__name__ + '.PassthruCallBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(PassthruCallBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        parser.add_argument('method', metavar='<method>', help=_('Vendor passthru method to be executed'))
        parser.add_argument('--arg', metavar='<key=value>', action='append', help=_('Argument to pass to the passthru method (repeat option to specify multiple arguments)'))
        parser.add_argument('--http-method', metavar='<http-method>', choices=v1_utils.HTTP_METHODS, default='POST', help=_('The HTTP method to use in the passthru request. One of %s. Defaults to POST.') % oscutils.format_list(v1_utils.HTTP_METHODS))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        arguments = utils.key_value_pairs_to_dict(parsed_args.arg)
        resp = baremetal_client.node.vendor_passthru(parsed_args.node, parsed_args.method, http_method=parsed_args.http_method, args=arguments)
        if resp:
            print(str(resp.to_dict()))