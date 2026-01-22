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
class ShowBaremetalNode(command.ShowOne):
    """Show baremetal node details"""
    log = logging.getLogger(__name__ + '.ShowBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node (or instance UUID if --instance is specified)'))
        parser.add_argument('--instance', dest='instance_uuid', action='store_true', default=False, help=_('<node> is an instance UUID.'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.NODE_DETAILED_RESOURCE.fields, default=[], help=_('One or more node fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        if parsed_args.instance_uuid:
            node = baremetal_client.node.get_by_instance_uuid(parsed_args.node, fields=fields)._info
        else:
            node = baremetal_client.node.get(parsed_args.node, fields=fields)._info
        node.pop('links', None)
        node.pop('ports', None)
        node.pop('portgroups', None)
        node.pop('states', None)
        node.pop('volume', None)
        if not fields or 'chassis_uuid' in fields:
            node.setdefault('chassis_uuid', '')
        return self.dict2columns(node)