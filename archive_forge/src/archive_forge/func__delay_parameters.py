import random
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_utils import timeutils
def _delay_parameters(self):
    """Return a tuple of the min delay and max jitter, in seconds."""
    min_wait_secs = self.properties[self.MIN_WAIT_SECS]
    max_jitter_secs = self.properties[self.MAX_JITTER] * self.properties[self.JITTER_MULTIPLIER_SECS]
    return (min_wait_secs, max_jitter_secs)