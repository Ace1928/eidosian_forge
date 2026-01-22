from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _has_state(self, state, raise_error=False):
    """ This function
        Args:
            state (NestedState): state to be tested
            raise_error (bool): whether ValueError should be raised when the state
                                is not registered
       Returns:
            bool: Whether state is registered in the machine
        Raises:
            ValueError: When raise_error is True and state is not registered
        """
    found = super(HierarchicalMachine, self)._has_state(state)
    if not found:
        for a_state in self.states:
            with self(a_state):
                if self._has_state(state):
                    return True
    if not found and raise_error:
        msg = 'State %s has not been added to the machine' % (state.name if hasattr(state, 'name') else state)
        raise ValueError(msg)
    return found