import logging
import asyncio
import contextvars
import inspect
from collections import deque
from functools import partial, reduce
import copy
from ..core import State, Condition, Transition, EventData, listify
from ..core import Event, MachineError, Machine
from .nesting import HierarchicalMachine, NestedState, NestedEvent, NestedTransition, resolve_order
class _DictionaryMock(dict):

    def __init__(self, item):
        super().__init__()
        self._value = item

    def __setitem__(self, key, item):
        self._value = item

    def __getitem__(self, key):
        return self._value

    def __repr__(self):
        return repr("{{'*': {0}}}".format(self._value))