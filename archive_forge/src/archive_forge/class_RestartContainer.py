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
class RestartContainer(command.Command):
    """Restart specified container"""
    log = logging.getLogger(__name__ + '.RestartContainer')

    def get_parser(self, prog_name):
        parser = super(RestartContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', nargs='+', help='ID or name of the (container)s to restart.')
        parser.add_argument('--timeout', metavar='<timeout>', default=10, help='Seconds to wait for stop before restarting (container)s')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        containers = parsed_args.container
        for container in containers:
            try:
                client.containers.restart(container, parsed_args.timeout)
                print(_('Request to restart container %s has been accepted') % container)
            except Exception as e:
                print('Restart for container %(container)s failed: %(e)s' % {'container': container, 'e': e})