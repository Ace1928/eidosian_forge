import argparse
from contextlib import closing
import io
import os
from oslo_log import log as logging
import tarfile
import time
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
from zunclient.i18n import _
class ActionList(command.Lister):
    """List actions on a container"""
    log = logging.getLogger(__name__ + '.ListActions')

    def get_parser(self, prog_name):
        parser = super(ActionList, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to list actions.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        container = parsed_args.container
        actions = client.actions.list(container)
        columns = ('user_id', 'container_uuid', 'request_id', 'action', 'message', 'start_time')
        return (columns, (utils.get_item_properties(action, columns) for action in actions))