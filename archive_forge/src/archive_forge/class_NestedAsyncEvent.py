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
class NestedAsyncEvent(NestedEvent):
    """ A collection of transitions assigned to the same trigger.
    This Event requires a (subclass of) `HierarchicalAsyncMachine`.
    """

    async def trigger_nested(self, event_data):
        """ Serially execute all transitions that match the current state,
        halting as soon as one successfully completes. NOTE: This should only
        be called by HierarchicalMachine instances.
        Args:
            event_data (AsyncEventData): The currently processed event.
        Returns: boolean indicating whether or not a transition was
            successfully executed (True if successful, False if not).
        """
        machine = event_data.machine
        model = event_data.model
        state_tree = machine.build_state_tree(getattr(model, machine.model_attribute), machine.state_cls.separator)
        state_tree = reduce(dict.get, machine.get_global_name(join=False), state_tree)
        ordered_states = resolve_order(state_tree)
        done = set()
        event_data.event = self
        for state_path in ordered_states:
            state_name = machine.state_cls.separator.join(state_path)
            if state_name not in done and state_name in self.transitions:
                event_data.state = machine.get_state(state_name)
                event_data.source_name = state_name
                event_data.source_path = copy.copy(state_path)
                await self._process(event_data)
                if event_data.result:
                    elems = state_path
                    while elems:
                        done.add(machine.state_cls.separator.join(elems))
                        elems.pop()
        return event_data.result

    async def _process(self, event_data):
        machine = event_data.machine
        await machine.callbacks(event_data.machine.prepare_event, event_data)
        _LOGGER.debug('%sExecuted machine preparation callbacks before conditions.', machine.name)
        for trans in self.transitions[event_data.source_name]:
            event_data.transition = trans
            event_data.result = await trans.execute(event_data)
            if event_data.result:
                break