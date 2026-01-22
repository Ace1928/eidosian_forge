import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def _eval_conditions(self, event_data):
    for cond in self.conditions:
        if not cond.check(event_data):
            _LOGGER.debug('%sTransition condition failed: %s() does not return %s. Transition halted.', event_data.machine.name, cond.func, cond.target)
            return False
    return True