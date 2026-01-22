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
class ListBIOSSettingBaremetalNode(command.Lister):
    """List a node's BIOS settings."""
    log = logging.getLogger(__name__ + '.ListBIOSSettingBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ListBIOSSettingBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        display_group = parser.add_mutually_exclusive_group(required=False)
        display_group.add_argument('--long', default=False, help=_('Show detailed information about the BIOS settings.'), action='store_true')
        display_group.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.BIOS_DETAILED_RESOURCE.fields, help=_("One or more node fields. Only these fields will be fetched from the server. Can not be used when '--long' is specified."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        labels = res_fields.BIOS_RESOURCE.labels
        fields = res_fields.BIOS_RESOURCE.fields
        params = {}
        if parsed_args.long:
            params['detail'] = parsed_args.long
            fields = res_fields.BIOS_DETAILED_RESOURCE.fields
            labels = res_fields.BIOS_DETAILED_RESOURCE.labels
        elif parsed_args.fields:
            params['detail'] = False
            fields = itertools.chain.from_iterable(parsed_args.fields)
            resource = res_fields.Resource(list(fields))
            fields = resource.fields
            labels = resource.labels
            params['fields'] = fields
        self.log.debug('params(%s)', params)
        baremetal_client = self.app.client_manager.baremetal
        settings = baremetal_client.node.list_bios_settings(parsed_args.node, **params)
        return (labels, (oscutils.get_dict_properties(s, fields) for s in settings))