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
class StatsContainer(command.ShowOne):
    """Display stats of the container."""
    log = logging.getLogger(__name__ + '.StatsContainer')

    def get_parser(self, prog_name):
        parser = super(StatsContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the (container)s to  display stats.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        container = parsed_args.container
        stats_info = client.containers.stats(container)
        return (stats_info.keys(), stats_info.values())