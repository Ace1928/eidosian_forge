import itertools
import time
from oslo_utils import timeutils
from taskflow.engines.action_engine import compiler as co
from taskflow import exceptions as exc
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
def _record_ending(self, timer, item_type, item_name, state):
    super(PrintingDurationListener, self)._record_ending(timer, item_type, item_name, state)
    self._printer("It took %s '%s' %0.2f seconds to finish." % (item_type, item_name, timer.elapsed()))