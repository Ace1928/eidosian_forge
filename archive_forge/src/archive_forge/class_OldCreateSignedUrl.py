import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class OldCreateSignedUrl(CreateSignedUrl):
    """Create a pre-signed url"""
    _description = _('Create a pre-signed url')
    deprecated = True
    log = logging.getLogger('deprecated')

    def take_action(self, parsed_args):
        self.log.warning(_('This command has been deprecated. Please use "messaging queue signed url" instead.'))
        return super(OldCreateSignedUrl, self).take_action(parsed_args)