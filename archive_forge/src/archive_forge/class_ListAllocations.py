import logging
from blazarclient import command
from blazarclient import utils
class ListAllocations(command.ListCommand):
    """List allocations for all resources of a type."""
    resource = 'allocation'
    log = logging.getLogger(__name__ + '.ListHostAllocations')
    list_columns = ['resource_id', 'reservations']

    def get_parser(self, prog_name):
        parser = super(ListAllocations, self).get_parser(prog_name)
        parser.add_argument('resource_type', choices=['host'], help='Show allocations for a resource type')
        parser.add_argument('--reservation-id', dest='reservation_id', default=None, help='Show only allocations with specific reservation_id')
        parser.add_argument('--lease-id', dest='lease_id', default=None, help='Show only allocations with specific lease_id')
        parser.add_argument('--sort-by', metavar='<allocation_column>', help='column name used to sort result', default='resource_id')
        return parser

    def get_data(self, parsed_args):
        self.log.debug('get_data(%s)' % parsed_args)
        data = self.retrieve_list(parsed_args)
        for resource in data:
            if parsed_args.lease_id is not None:
                resource['reservations'] = list(filter(lambda d: d['lease_id'] == parsed_args.lease_id, resource['reservations']))
            if parsed_args.reservation_id is not None:
                resource['reservations'] = list(filter(lambda d: d['id'] == parsed_args.reservation_id, resource['reservations']))
        return self.setup_columns(data, parsed_args)

    def args2body(self, parsed_args):
        params = super(ListAllocations, self).args2body(parsed_args)
        if parsed_args.resource_type == 'host':
            params.update(dict(resource='os-hosts'))
        return params