import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class ListQueues(command.Lister):
    """List available queues"""
    _description = _('List available queues')
    log = logging.getLogger(__name__ + '.ListQueues')

    def get_parser(self, prog_name):
        parser = super(ListQueues, self).get_parser(prog_name)
        parser.add_argument('--marker', metavar='<queue_id>', help="Queue's paging marker")
        parser.add_argument('--limit', metavar='<limit>', help='Page size limit')
        parser.add_argument('--detailed', action='store_true', help='If show detailed information of queue')
        parser.add_argument('--with_count', action='store_true', help='If show amount information of queue')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        kwargs = {}
        columns = ['Name']
        if parsed_args.marker is not None:
            kwargs['marker'] = parsed_args.marker
        if parsed_args.limit is not None:
            kwargs['limit'] = parsed_args.limit
        if parsed_args.detailed is not None and parsed_args.detailed:
            kwargs['detailed'] = parsed_args.detailed
            columns.extend(['Metadata_Dict', 'Href'])
        if parsed_args.with_count is not None and parsed_args.with_count:
            kwargs['with_count'] = parsed_args.with_count
        data, count = client.queues(**kwargs)
        if count:
            print('Queues in total: %s' % count)
        columns = tuple(columns)
        return (columns, (utils.get_item_properties(s, columns) for s in data))