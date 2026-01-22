import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class OldDeleteSubscription(DeleteSubscription):
    """Delete a subscription"""
    _description = _('Delete a subscription')
    deprecated = True
    log = logging.getLogger('deprecated')

    def take_action(self, parsed_args):
        self.log.warning(_('This command has been deprecated. Please use "messaging subscription delete" instead.'))
        return super(OldDeleteSubscription, self).take_action(parsed_args)