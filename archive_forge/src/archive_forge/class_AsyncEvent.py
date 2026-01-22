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
class AsyncEvent(Event):
    """ A collection of transitions assigned to the same trigger """

    async def trigger(self, model, *args, **kwargs):
        """ Serially execute all transitions that match the current state,
        halting as soon as one successfully completes. Note that `AsyncEvent` triggers must be awaited.
        Args:
            args and kwargs: Optional positional or named arguments that will
                be passed onto the EventData object, enabling arbitrary state
                information to be passed on to downstream triggered functions.
        Returns: boolean indicating whether or not a transition was
            successfully executed (True if successful, False if not).
        """
        func = partial(self._trigger, EventData(None, self, self.machine, model, args=args, kwargs=kwargs))
        return await self.machine.process_context(func, model)

    async def _trigger(self, event_data):
        event_data.state = self.machine.get_state(getattr(event_data.model, self.machine.model_attribute))
        try:
            if self._is_valid_source(event_data.state):
                await self._process(event_data)
        except Exception as err:
            _LOGGER.error('%sException was raised while processing the trigger: %s', self.machine.name, err)
            event_data.error = err
            if self.machine.on_exception:
                await self.machine.callbacks(self.machine.on_exception, event_data)
            else:
                raise
        finally:
            await self.machine.callbacks(self.machine.finalize_event, event_data)
            _LOGGER.debug('%sExecuted machine finalize callbacks', self.machine.name)
        return event_data.result

    async def _process(self, event_data):
        await self.machine.callbacks(self.machine.prepare_event, event_data)
        _LOGGER.debug('%sExecuted machine preparation callbacks before conditions.', self.machine.name)
        for trans in self.transitions[event_data.state.name]:
            event_data.transition = trans
            event_data.result = await trans.execute(event_data)
            if event_data.result:
                break