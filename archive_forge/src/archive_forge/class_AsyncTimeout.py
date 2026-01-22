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
class AsyncTimeout(AsyncState):
    """
    Adds timeout functionality to an asynchronous state. Timeouts are handled model-specific.

    Attributes:
        timeout (float): Seconds after which a timeout function should be
                         called.
        on_timeout (list): Functions to call when a timeout is triggered.
        runner (dict): Keeps track of running timeout tasks to cancel when a state is exited.
    """
    dynamic_methods = ['on_timeout']

    def __init__(self, *args, **kwargs):
        """
        Args:
            **kwargs: If kwargs contain 'timeout', assign the float value to
                self.timeout. If timeout is set, 'on_timeout' needs to be
                passed with kwargs as well or an AttributeError will be thrown
                if timeout is not passed or equal 0.
        """
        self.timeout = kwargs.pop('timeout', 0)
        self._on_timeout = None
        if self.timeout > 0:
            try:
                self.on_timeout = kwargs.pop('on_timeout')
            except KeyError:
                raise AttributeError("Timeout state requires 'on_timeout' when timeout is set.") from None
        else:
            self.on_timeout = kwargs.pop('on_timeout', None)
        self.runner = {}
        super().__init__(*args, **kwargs)

    async def enter(self, event_data):
        """
        Extends `transitions.core.State.enter` by starting a timeout timer for
        the current model when the state is entered and self.timeout is larger
        than 0.

        Args:
            event_data (EventData): events representing the currently processed event.
        """
        if self.timeout > 0:
            self.runner[id(event_data.model)] = self.create_timer(event_data)
        await super().enter(event_data)

    async def exit(self, event_data):
        """
        Cancels running timeout tasks stored in `self.runner` first (when not note) before
        calling further exit callbacks.

        Args:
            event_data (EventData): Data representing the currently processed event.

        Returns:

        """
        timer_task = self.runner.get(id(event_data.model), None)
        if timer_task is not None and (not timer_task.done()):
            timer_task.cancel()
        await super().exit(event_data)

    def create_timer(self, event_data):
        """
        Creates and returns a running timer. Shields self._process_timeout to prevent cancellation when
        transitioning away from the current state (which cancels the timer) while processing timeout callbacks.
        Args:
            event_data (EventData): Data representing the currently processed event.

        Returns (cancellable): A running timer with a cancel method
        """

        async def _timeout():
            try:
                await asyncio.sleep(self.timeout)
                await asyncio.shield(self._process_timeout(event_data))
            except asyncio.CancelledError:
                pass
        return asyncio.ensure_future(_timeout())

    async def _process_timeout(self, event_data):
        _LOGGER.debug('%sTimeout state %s. Processing callbacks...', event_data.machine.name, self.name)
        await event_data.machine.callbacks(self.on_timeout, event_data)
        _LOGGER.info('%sTimeout state %s processed.', event_data.machine.name, self.name)

    @property
    def on_timeout(self):
        """
        List of strings and callables to be called when the state timeouts.
        """
        return self._on_timeout

    @on_timeout.setter
    def on_timeout(self, value):
        """ Listifies passed values and assigns them to on_timeout."""
        self._on_timeout = listify(value)