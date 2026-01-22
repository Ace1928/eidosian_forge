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
class NestedAsyncState(NestedState, AsyncState):
    """ A state that allows substates. Callback execution is done asynchronously. """

    async def scoped_enter(self, event_data, scope=None):
        self._scope = scope or []
        await self.enter(event_data)
        self._scope = []

    async def scoped_exit(self, event_data, scope=None):
        self._scope = scope or []
        await self.exit(event_data)
        self._scope = []