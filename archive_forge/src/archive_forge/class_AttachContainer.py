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
class AttachContainer(command.Command):
    """Attach to a running container"""
    log = logging.getLogger(__name__ + '.AttachContainer')

    def get_parser(self, prog_name):
        parser = super(AttachContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to be attached to.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        response = client.containers.attach(parsed_args.container)
        websocketclient.do_attach(client, response, parsed_args.container, '~', 0.5)