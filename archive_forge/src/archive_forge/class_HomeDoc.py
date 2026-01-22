import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class HomeDoc(command.Command):
    """Display the resource doc of Zaqar server"""
    _description = _('Display detailed resource doc of Zaqar server')
    log = logging.getLogger(__name__ + '.HomeDoc')

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        homedoc = client.homedoc()
        print(json.dumps(homedoc, indent=4, sort_keys=True))