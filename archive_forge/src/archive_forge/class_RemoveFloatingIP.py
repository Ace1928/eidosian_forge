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
class RemoveFloatingIP(command.Command):
    """Remove floating IP address from container"""
    log = logging.getLogger(__name__ + '.RemoveFloatingIP')

    def get_parser(self, prog_name):
        parser = super(RemoveFloatingIP, self).get_parser(prog_name)
        parser.add_argument('ip_address', metavar='<ip-address>', help=_('Floating IP address to remove from container (IP only)'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        attrs = {}
        obj = network_client.find_ip(parsed_args.ip_address, ignore_missing=False)
        attrs['port_id'] = None
        network_client.update_ip(obj, **attrs)