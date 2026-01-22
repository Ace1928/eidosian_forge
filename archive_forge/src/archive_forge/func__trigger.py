import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def _trigger(self, event_data):
    """ Internal trigger function called by the ``Machine`` instance. This should not
        be called directly but via the public method ``Machine.process``.
        Args:
            event_data (EventData): The currently processed event. State, result and (potentially) error might be
            overridden.
        Returns: boolean indicating whether a transition was
            successfully executed (True if successful, False if not).
        """
    event_data.state = self.machine.get_model_state(event_data.model)
    try:
        if self._is_valid_source(event_data.state):
            self._process(event_data)
    except Exception as err:
        event_data.error = err
        if self.machine.on_exception:
            self.machine.callbacks(self.machine.on_exception, event_data)
        else:
            raise
    finally:
        try:
            self.machine.callbacks(self.machine.finalize_event, event_data)
            _LOGGER.debug('%sExecuted machine finalize callbacks', self.machine.name)
        except Exception as err:
            _LOGGER.error('%sWhile executing finalize callbacks a %s occurred: %s.', self.machine.name, type(err).__name__, str(err))
    return event_data.result