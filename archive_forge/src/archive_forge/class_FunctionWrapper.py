from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
class FunctionWrapper(object):
    """ A wrapper to enable transitions' convenience function to_<state> for nested states.
        This allows to call model.to_A.s1.C() in case a custom separator has been chosen."""

    def __init__(self, func, path):
        """
        Args:
            func: Function to be called at the end of the path.
            path: If path is an empty string, assign function
        """
        if path:
            self.add(func, path)
            self._func = None
        else:
            self._func = func

    def add(self, func, path):
        """ Assigns a `FunctionWrapper` as an attribute named like the next segment of the substates
            path.
        Args:
            func (callable): Function to be called at the end of the path.
            path (list of strings): Remaining segment of the substate path.
        """
        if path:
            name = path[0]
            if name[0].isdigit():
                name = 's' + name
            if hasattr(self, name):
                getattr(self, name).add(func, path[1:])
            else:
                setattr(self, name, FunctionWrapper(func, path[1:]))
        else:
            self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)