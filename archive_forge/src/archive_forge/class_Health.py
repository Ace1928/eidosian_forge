import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class Health(command.Command):
    """Display detailed health status of Zaqar server"""
    _description = _('Display detailed health status of Zaqar server')
    log = logging.getLogger(__name__ + '.Health')

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        health = client.health()
        print(json.dumps(health, indent=4, sort_keys=True))