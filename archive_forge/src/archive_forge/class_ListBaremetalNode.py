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
class ListBaremetalNode(command.Lister):
    """List baremetal nodes"""
    log = logging.getLogger(__name__ + '.ListBaremetalNode')
    PROVISION_STATES = ['active', 'deleted', 'rebuild', 'inspect', 'provide', 'manage', 'clean', 'adopt', 'abort']

    def get_parser(self, prog_name):
        parser = super(ListBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--limit', metavar='<limit>', type=int, help=_('Maximum number of nodes to return per request, 0 for no limit. Default is the maximum number used by the Baremetal API Service.'))
        parser.add_argument('--marker', metavar='<node>', help=_('Node UUID (for example, of the last node in the list from a previous request). Returns the list of nodes after this UUID.'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', help=_('Sort output by specified node fields and directions (asc or desc) (default: asc). Multiple fields and directions can be specified, separated by comma.'))
        maint_group = parser.add_mutually_exclusive_group(required=False)
        maint_group.add_argument('--maintenance', dest='maintenance', action='store_true', default=None, help=_('Limit list to nodes in maintenance mode'))
        maint_group.add_argument('--no-maintenance', dest='maintenance', action='store_false', default=None, help=_('Limit list to nodes not in maintenance mode'))
        retired_group = parser.add_mutually_exclusive_group(required=False)
        retired_group.add_argument('--retired', dest='retired', action='store_true', default=None, help=_('Limit list to retired nodes.'))
        retired_group.add_argument('--no-retired', dest='retired', action='store_false', default=None, help=_('Limit list to not retired nodes.'))
        parser.add_argument('--fault', dest='fault', metavar='<fault>', help=_('List nodes in specified fault.'))
        associated_group = parser.add_mutually_exclusive_group()
        associated_group.add_argument('--associated', action='store_true', help=_('List only nodes associated with an instance.'))
        associated_group.add_argument('--unassociated', action='store_true', help=_('List only nodes not associated with an instance.'))
        parser.add_argument('--provision-state', dest='provision_state', metavar='<provision state>', help=_('List nodes in specified provision state.'))
        parser.add_argument('--driver', dest='driver', metavar='<driver>', help=_('Limit list to nodes with driver <driver>'))
        parser.add_argument('--resource-class', dest='resource_class', metavar='<resource class>', help=_('Limit list to nodes with resource class <resource class>'))
        parser.add_argument('--conductor-group', metavar='<conductor_group>', help=_('Limit list to nodes with conductor group <conductor group>'))
        parser.add_argument('--conductor', metavar='<conductor>', help=_('Limit list to nodes with conductor <conductor>'))
        parser.add_argument('--chassis', dest='chassis', metavar='<chassis UUID>', help=_('Limit list to nodes of this chassis'))
        parser.add_argument('--owner', metavar='<owner>', help=_('Limit list to nodes with owner <owner>'))
        parser.add_argument('--lessee', metavar='<lessee>', help=_('Limit list to nodes with lessee <lessee>'))
        parser.add_argument('--description-contains', metavar='<description_contains>', help=_('Limit list to nodes with description contains <description_contains>'))
        sharded_group = parser.add_mutually_exclusive_group(required=False)
        sharded_group.add_argument('--sharded', dest='sharded', help=_('List only nodes that are sharded.'), default=None, action='store_true')
        sharded_group.add_argument('--unsharded', dest='sharded', help=_('List only nodes that are not sharded.'), default=None, action='store_false')
        parser.add_argument('--shards', nargs='+', metavar='<shards>', help=_('List only nodes that are in shards <shards>.'))
        display_group = parser.add_mutually_exclusive_group(required=False)
        display_group.add_argument('--long', default=False, help=_('Show detailed information about the nodes.'), action='store_true')
        display_group.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.NODE_DETAILED_RESOURCE.fields, help=_("One or more node fields. Only these fields will be fetched from the server. Can not be used when '--long' is specified."))
        children_group = parser.add_mutually_exclusive_group(required=False)
        children_group.add_argument('--include-children', action='store_true', help=_('Include children in the node list.'))
        children_group.add_argument('--parent-node', dest='parent_node', metavar='<parent_node>', help=_('List only nodes associated with a parent node.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.baremetal
        columns = res_fields.NODE_RESOURCE.fields
        labels = res_fields.NODE_RESOURCE.labels
        params = {}
        if parsed_args.limit is not None and parsed_args.limit < 0:
            raise exc.CommandError(_('Expected non-negative --limit, got %s') % parsed_args.limit)
        params['limit'] = parsed_args.limit
        params['marker'] = parsed_args.marker
        if parsed_args.associated:
            params['associated'] = True
        if parsed_args.unassociated:
            params['associated'] = False
        for field in ['maintenance', 'fault', 'conductor_group', 'retired', 'sharded']:
            if getattr(parsed_args, field) is not None:
                params[field] = getattr(parsed_args, field)
        for field in ['provision_state', 'driver', 'resource_class', 'chassis', 'conductor', 'owner', 'lessee', 'description_contains', 'shards', 'parent_node']:
            if getattr(parsed_args, field):
                params[field] = getattr(parsed_args, field)
        if parsed_args.include_children:
            params['include_children'] = True
        if parsed_args.long:
            params['detail'] = parsed_args.long
            columns = res_fields.NODE_DETAILED_RESOURCE.fields
            labels = res_fields.NODE_DETAILED_RESOURCE.labels
        elif parsed_args.fields:
            params['detail'] = False
            fields = itertools.chain.from_iterable(parsed_args.fields)
            resource = res_fields.Resource(list(fields))
            columns = resource.fields
            labels = resource.labels
            params['fields'] = columns
        self.log.debug('params(%s)', params)
        data = client.node.list(**params)
        data = oscutils.sort_items(data, parsed_args.sort)
        return (labels, (oscutils.get_item_properties(s, columns, formatters={'Properties': oscutils.format_dict}) for s in data))