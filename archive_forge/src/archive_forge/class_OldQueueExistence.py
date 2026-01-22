import json
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.transport import errors
class OldQueueExistence(CheckQueueExistence):
    """Check queue existence"""
    _description = _('Check queue existence')
    deprecated = True
    log = logging.getLogger('deprecated')

    def take_action(self, parsed_args):
        self.log.warning(_('This command has been deprecated. Please use "messaging queue exists" instead.'))
        return super(OldQueueExistence, self).take_action(parsed_args)