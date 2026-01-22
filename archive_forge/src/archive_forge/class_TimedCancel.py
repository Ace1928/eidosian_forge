import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
class TimedCancel(Timeout):

    def trigger(self, generator):
        """Trigger the timeout on a given generator."""
        generator.close()
        return False