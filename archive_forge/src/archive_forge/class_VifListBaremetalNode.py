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
class VifListBaremetalNode(command.Lister):
    """Show attached VIFs for a node"""
    log = logging.getLogger(__name__ + '.VifListBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(VifListBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        columns = res_fields.VIF_RESOURCE.fields
        labels = res_fields.VIF_RESOURCE.labels
        baremetal_client = self.app.client_manager.baremetal
        data = baremetal_client.node.vif_list(parsed_args.node)
        return (labels, (oscutils.get_item_properties(s, columns) for s in data))