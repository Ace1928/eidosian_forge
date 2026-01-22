from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
def _add_model_to_state(self, state, model):
    super(LockedMachine, self)._add_model_to_state(state, model)
    for prefix in self.state_cls.dynamic_methods:
        callback = '{0}_{1}'.format(prefix, self._get_qualified_state_name(state))
        func = getattr(model, callback, None)
        if isinstance(func, partial) and func.func != state.add_callback:
            state.add_callback(prefix[3:], callback)