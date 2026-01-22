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
class NestedAsyncTransition(AsyncTransition, NestedTransition):
    """ Representation of an asynchronous transition managed by a ``HierarchicalMachine`` instance. """

    async def _change_state(self, event_data):
        if hasattr(event_data.machine, 'model_graphs'):
            graph = event_data.machine.model_graphs[id(event_data.model)]
            graph.reset_styling()
            graph.set_previous_transition(self.source, self.dest)
        state_tree, exit_partials, enter_partials = self._resolve_transition(event_data)
        for func in exit_partials:
            await func()
        self._update_model(event_data, state_tree)
        for func in enter_partials:
            await func()