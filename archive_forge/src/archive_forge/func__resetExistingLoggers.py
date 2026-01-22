import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
def _resetExistingLoggers(parent='root'):
    """Reset the logger named 'parent' and all its children to their initial
    state, if they already exist in the current configuration.
    """
    root = logging.root
    existing = sorted(root.manager.loggerDict.keys())
    if parent == 'root':
        loggers_to_reset = [parent] + existing
    elif parent not in existing:
        return
    elif parent in existing:
        loggers_to_reset = [parent]
        i = existing.index(parent) + 1
        prefixed = parent + '.'
        pflen = len(prefixed)
        num_existing = len(existing)
        while i < num_existing:
            if existing[i][:pflen] == prefixed:
                loggers_to_reset.append(existing[i])
            i += 1
    for name in loggers_to_reset:
        if name == 'root':
            root.setLevel(logging.WARNING)
            for h in root.handlers[:]:
                root.removeHandler(h)
            for f in root.filters[:]:
                root.removeFilters(f)
            root.disabled = False
        else:
            logger = root.manager.loggerDict[name]
            logger.level = logging.NOTSET
            logger.handlers = []
            logger.filters = []
            logger.propagate = True
            logger.disabled = False