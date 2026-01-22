import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ListBaremetalAllocation(command.Lister):
    """List baremetal allocations."""
    log = logging.getLogger(__name__ + '.ListBaremetalAllocation')

    def get_parser(self, prog_name):
        parser = super(ListBaremetalAllocation, self).get_parser(prog_name)
        parser.add_argument('--limit', metavar='<limit>', type=int, help=_('Maximum number of allocations to return per request, 0 for no limit. Default is the maximum number used by the Baremetal API Service.'))
        parser.add_argument('--marker', metavar='<allocation>', help=_('Allocation UUID (for example, of the last allocation in the list from a previous request). Returns the list of allocations after this UUID.'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', help=_('Sort output by specified allocation fields and directions (asc or desc) (default: asc). Multiple fields and directions can be specified, separated by comma.'))
        parser.add_argument('--node', metavar='<node>', help=_('Only list allocations of this node (name or UUID).'))
        parser.add_argument('--resource-class', metavar='<resource_class>', help=_('Only list allocations with this resource class.'))
        parser.add_argument('--state', metavar='<state>', help=_('Only list allocations in this state.'))
        parser.add_argument('--owner', metavar='<owner>', help=_('Only list allocations with this owner.'))
        display_group = parser.add_mutually_exclusive_group(required=False)
        display_group.add_argument('--long', default=False, help=_('Show detailed information about the allocations.'), action='store_true')
        display_group.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.ALLOCATION_DETAILED_RESOURCE.fields, help=_("One or more allocation fields. Only these fields will be fetched from the server. Can not be used when '--long' is specified."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.baremetal
        params = {}
        if parsed_args.limit is not None and parsed_args.limit < 0:
            raise exc.CommandError(_('Expected non-negative --limit, got %s') % parsed_args.limit)
        params['limit'] = parsed_args.limit
        params['marker'] = parsed_args.marker
        for field in ('node', 'resource_class', 'state', 'owner'):
            value = getattr(parsed_args, field)
            if value is not None:
                params[field] = value
        if parsed_args.long:
            columns = res_fields.ALLOCATION_DETAILED_RESOURCE.fields
            labels = res_fields.ALLOCATION_DETAILED_RESOURCE.labels
        elif parsed_args.fields:
            fields = itertools.chain.from_iterable(parsed_args.fields)
            resource = res_fields.Resource(list(fields))
            columns = resource.fields
            labels = resource.labels
            params['fields'] = columns
        else:
            columns = res_fields.ALLOCATION_RESOURCE.fields
            labels = res_fields.ALLOCATION_RESOURCE.labels
        self.log.debug('params(%s)', params)
        data = client.allocation.list(**params)
        data = oscutils.sort_items(data, parsed_args.sort)
        return (labels, (oscutils.get_item_properties(s, columns) for s in data))