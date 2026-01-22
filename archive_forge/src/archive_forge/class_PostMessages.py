import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class PostMessages(command.Command):
    """Post messages for a given queue"""
    _description = _('Post messages for a given queue')
    log = logging.getLogger(__name__ + '.PostMessages')

    def get_parser(self, prog_name):
        parser = super(PostMessages, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        parser.add_argument('messages', type=json.loads, metavar='<messages>', help='Messages to be posted.')
        parser.add_argument('--client-id', metavar='<client_id>', default=os.environ.get('OS_MESSAGE_CLIENT_ID'), help='A UUID for each client instance.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        if not parsed_args.client_id:
            raise AttributeError('<--client-id> option is missing and environment variable OS_MESSAGE_CLIENT_ID is not set. Please at least either pass in the client id or set the environment variable')
        else:
            client.client_uuid = parsed_args.client_id
        queue = client.queue(parsed_args.queue_name)
        queue.post(parsed_args.messages)