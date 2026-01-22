import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class ListMessages(command.Lister):
    """List all messages for a given queue"""
    _description = _('List all messages for a given queue')
    log = logging.getLogger(__name__ + '.ListMessages')

    def get_parser(self, prog_name):
        parser = super(ListMessages, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        parser.add_argument('--message-ids', metavar='<message_ids>', help="List of messages' ids to retrieve")
        parser.add_argument('--limit', metavar='<limit>', type=int, help='Maximum number of messages to get')
        parser.add_argument('--echo', action='store_true', help="Whether to get this client's own messages")
        parser.add_argument('--include-claimed', action='store_true', help='Whether to include claimed messages')
        parser.add_argument('--include-delayed', action='store_true', help='Whether to include delayed messages')
        parser.add_argument('--client-id', metavar='<client_id>', default=os.environ.get('OS_MESSAGE_CLIENT_ID'), help='A UUID for each client instance.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        if not parsed_args.client_id:
            raise AttributeError('<--client-id> option is missing and environment variable OS_MESSAGE_CLIENT_ID is not set. Please at least either pass in the client id or set the environment variable')
        else:
            client.client_uuid = parsed_args.client_id
        kwargs = {}
        if parsed_args.limit is not None:
            kwargs['limit'] = parsed_args.limit
        if parsed_args.echo is not None:
            kwargs['echo'] = parsed_args.echo
        if parsed_args.include_claimed is not None:
            kwargs['include_claimed'] = parsed_args.include_claimed
        if parsed_args.include_delayed is not None:
            kwargs['include_delayed'] = parsed_args.include_delayed
        queue = client.queue(parsed_args.queue_name)
        if parsed_args.message_ids:
            messages = queue.messages(parsed_args.message_ids.split(','), **kwargs)
        else:
            messages = queue.messages(**kwargs)
        columns = ('ID', 'Body', 'TTL', 'Age', 'Claim ID', 'Checksum')
        return (columns, (utils.get_item_properties(s, columns) for s in messages))