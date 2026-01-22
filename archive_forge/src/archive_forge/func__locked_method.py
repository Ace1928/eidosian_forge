from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
def _locked_method(self, func, *args, **kwargs):
    if self._ident.current != get_ident():
        with nested(*self.machine_context):
            return func(*args, **kwargs)
    else:
        return func(*args, **kwargs)