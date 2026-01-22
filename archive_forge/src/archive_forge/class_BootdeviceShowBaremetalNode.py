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
class BootdeviceShowBaremetalNode(command.ShowOne):
    """Show the boot device information for a node"""
    log = logging.getLogger(__name__ + '.BootdeviceShowBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(BootdeviceShowBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        parser.add_argument('--supported', dest='supported', action='store_true', default=False, help=_('Show the supported boot devices'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.supported:
            info = baremetal_client.node.get_supported_boot_devices(parsed_args.node)
            boot_device_list = info.get('supported_boot_devices', [])
            info['supported_boot_devices'] = ', '.join(boot_device_list)
        else:
            info = baremetal_client.node.get_boot_device(parsed_args.node)
        return zip(*sorted(info.items()))