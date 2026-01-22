import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@after_state_change.setter
def after_state_change(self, value):
    self._after_state_change = listify(value)