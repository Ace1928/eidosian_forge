import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def _checked_assignment(self, model, name, func):
    if hasattr(model, name):
        _LOGGER.warning("%sModel already contains an attribute '%s'. Skip binding.", self.name, name)
    else:
        setattr(model, name, func)