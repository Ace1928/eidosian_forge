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
class PowerBaremetalNode(command.Command):
    """Base power state class, for setting the power of a node"""
    log = logging.getLogger(__name__ + '.PowerBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(PowerBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes."))
        parser.add_argument('--power-timeout', metavar='<power-timeout>', default=None, type=int, help=_('Timeout (in seconds, positive integer) to wait for the target power state before erroring out.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        soft = getattr(parsed_args, 'soft', False)
        for node in parsed_args.nodes:
            baremetal_client.node.set_power_state(node, self.POWER_STATE, soft, timeout=parsed_args.power_timeout)