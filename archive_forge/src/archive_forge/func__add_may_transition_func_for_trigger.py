import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def _add_may_transition_func_for_trigger(self, trigger, model):
    self._checked_assignment(model, 'may_%s' % trigger, partial(self._can_trigger, model, trigger))