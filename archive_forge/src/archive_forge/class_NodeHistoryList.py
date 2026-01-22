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
class NodeHistoryList(command.Lister):
    """Get history events for a baremetal node."""
    log = logging.getLogger(__name__ + '.NodeHistoryList')

    def get_parser(self, prog_name):
        parser = super(NodeHistoryList, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node.'))
        parser.add_argument('--long', default=False, help=_('Show detailed information about the node history events.'), action='store_true')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.long:
            labels = res_fields.NODE_HISTORY_DETAILED_RESOURCE.labels
            fields = res_fields.NODE_HISTORY_DETAILED_RESOURCE.fields
        else:
            labels = res_fields.NODE_HISTORY_RESOURCE.labels
            fields = res_fields.NODE_HISTORY_RESOURCE.fields
        data = baremetal_client.node.get_history_list(parsed_args.node, parsed_args.long)
        return (labels, (oscutils.get_dict_properties(s, fields) for s in data))