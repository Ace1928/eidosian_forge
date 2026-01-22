import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class CheckStack(StackActionBase):
    """Check a stack."""
    log = logging.getLogger(__name__ + '.CheckStack')

    def get_parser(self, prog_name):
        return self._get_parser(prog_name, _('Stack(s) to check update (name or ID)'), _('Wait for check to complete'))

    def take_action(self, parsed_args):
        return self._take_action(parsed_args, self.app.client_manager.orchestration.actions.check, 'CHECK')