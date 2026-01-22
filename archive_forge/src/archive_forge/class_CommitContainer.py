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
class CommitContainer(command.Command):
    """Create a new image from a container's changes"""
    log = logging.getLogger(__name__ + '.CommitContainer')

    def get_parser(self, prog_name):
        parser = super(CommitContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the (container)s to commit.')
        parser.add_argument('repository', metavar='<repository>[:<tag>]', help='Repository and tag of the new image.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        container = parsed_args.container
        opts = zun_utils.check_commit_container_args(parsed_args)
        opts = zun_utils.remove_null_parms(**opts)
        try:
            image = client.containers.commit(container, **opts)
            print('Request to commit container %s has been accepted. The image is %s.' % (container, image['uuid']))
        except Exception as e:
            print('commit container %(container)s failed: %(e)s' % {'container': container, 'e': e})