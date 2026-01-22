import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
def _status_ok(self, status):
    return status in self.WAIT_STATUSES