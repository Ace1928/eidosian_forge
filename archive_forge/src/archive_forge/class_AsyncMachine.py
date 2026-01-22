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
class AsyncMachine(Machine):
    """ Machine manages states, transitions and models. In case it is initialized without a specific model
    (or specifically no model), it will also act as a model itself. Machine takes also care of decorating
    models with conveniences functions related to added transitions and states during runtime.

    Attributes:
        states (OrderedDict): Collection of all registered states.
        events (dict): Collection of transitions ordered by trigger/event.
        models (list): List of models attached to the machine.
        initial (str): Name of the initial state for new models.
        prepare_event (list): Callbacks executed when an event is triggered.
        before_state_change (list): Callbacks executed after condition checks but before transition is conducted.
            Callbacks will be executed BEFORE the custom callbacks assigned to the transition.
        after_state_change (list): Callbacks executed after the transition has been conducted.
            Callbacks will be executed AFTER the custom callbacks assigned to the transition.
        finalize_event (list): Callbacks will be executed after all transitions callbacks have been executed.
            Callbacks mentioned here will also be called if a transition or condition check raised an error.
        on_exception: A callable called when an event raises an exception. If not set,
            the Exception will be raised instead.
        queued (bool or str): Whether transitions in callbacks should be executed immediately (False) or sequentially.
        send_event (bool): When True, any arguments passed to trigger methods will be wrapped in an EventData
            object, allowing indirect and encapsulated access to data. When False, all positional and keyword
            arguments will be passed directly to all callback methods.
        auto_transitions (bool):  When True (default), every state will automatically have an associated
            to_{state}() convenience trigger in the base model.
        ignore_invalid_triggers (bool): When True, any calls to trigger methods that are not valid for the
            present state (e.g., calling an a_to_b() trigger when the current state is c) will be silently
            ignored rather than raising an invalid transition exception.
        name (str): Name of the ``Machine`` instance mainly used for easier log message distinction.
    """
    state_cls = AsyncState
    transition_cls = AsyncTransition
    event_cls = AsyncEvent
    async_tasks = {}
    protected_tasks = []
    current_context = contextvars.ContextVar('current_context', default=None)

    def __init__(self, model=Machine.self_literal, states=None, initial='initial', transitions=None, send_event=False, auto_transitions=True, ordered_transitions=False, ignore_invalid_triggers=None, before_state_change=None, after_state_change=None, name=None, queued=False, prepare_event=None, finalize_event=None, model_attribute='state', on_exception=None, **kwargs):
        self._transition_queue_dict = {}
        super().__init__(model=model, states=states, initial=initial, transitions=transitions, send_event=send_event, auto_transitions=auto_transitions, ordered_transitions=ordered_transitions, ignore_invalid_triggers=ignore_invalid_triggers, before_state_change=before_state_change, after_state_change=after_state_change, name=name, queued=queued, prepare_event=prepare_event, finalize_event=finalize_event, model_attribute=model_attribute, on_exception=on_exception, **kwargs)
        if self.has_queue is True:
            self._transition_queue_dict = _DictionaryMock(self._transition_queue)

    def add_model(self, model, initial=None):
        super().add_model(model, initial)
        if self.has_queue == 'model':
            for mod in listify(model):
                self._transition_queue_dict[id(self) if mod is self.self_literal else id(mod)] = deque()

    async def dispatch(self, trigger, *args, **kwargs):
        """ Trigger an event on all models assigned to the machine.
        Args:
            trigger (str): Event name
            *args (list): List of arguments passed to the event trigger
            **kwargs (dict): Dictionary of keyword arguments passed to the event trigger
        Returns:
            bool The truth value of all triggers combined with AND
        """
        results = await self.await_all([partial(getattr(model, trigger), *args, **kwargs) for model in self.models])
        return all(results)

    async def callbacks(self, funcs, event_data):
        """ Triggers a list of callbacks """
        await self.await_all([partial(event_data.machine.callback, func, event_data) for func in funcs])

    async def callback(self, func, event_data):
        """ Trigger a callback function with passed event_data parameters. In case func is a string,
            the callable will be resolved from the passed model in event_data. This function is not intended to
            be called directly but through state and transition callback definitions.
        Args:
            func (string, callable): The callback function.
                1. First, if the func is callable, just call it
                2. Second, we try to import string assuming it is a path to a func
                3. Fallback to a model attribute
            event_data (EventData): An EventData instance to pass to the
                callback (if event sending is enabled) or to extract arguments
                from (if event sending is disabled).
        """
        func = self.resolve_callable(func, event_data)
        res = func(event_data) if self.send_event else func(*event_data.args, **event_data.kwargs)
        if inspect.isawaitable(res):
            await res

    @staticmethod
    async def await_all(callables):
        """
        Executes callables without parameters in parallel and collects their results.
        Args:
            callables (list): A list of callable functions

        Returns:
            list: A list of results. Using asyncio the list will be in the same order as the passed callables.
        """
        return await asyncio.gather(*[func() for func in callables])

    async def switch_model_context(self, model):
        """
        This method is called by an `AsyncTransition` when all conditional tests have passed
        and the transition will happen. This requires already running tasks to be cancelled.
        Args:
            model (object): The currently processed model
        """
        for running_task in self.async_tasks.get(id(model), []):
            if self.current_context.get() == running_task or running_task in self.protected_tasks:
                continue
            if running_task.done() is False:
                _LOGGER.debug('Cancel running tasks...')
                running_task.cancel()

    async def process_context(self, func, model):
        """
        This function is called by an `AsyncEvent` to make callbacks processed in Event._trigger cancellable.
        Using asyncio this will result in a try-catch block catching CancelledEvents.
        Args:
            func (partial): The partial of Event._trigger with all parameters already assigned
            model (object): The currently processed model

        Returns:
            bool: returns the success state of the triggered event
        """
        if self.current_context.get() is None:
            self.current_context.set(asyncio.current_task())
            if id(model) in self.async_tasks:
                self.async_tasks[id(model)].append(asyncio.current_task())
            else:
                self.async_tasks[id(model)] = [asyncio.current_task()]
            try:
                res = await self._process_async(func, model)
            except asyncio.CancelledError:
                res = False
            finally:
                self.async_tasks[id(model)].remove(asyncio.current_task())
                if len(self.async_tasks[id(model)]) == 0:
                    del self.async_tasks[id(model)]
        else:
            res = await self._process_async(func, model)
        return res

    def remove_model(self, model):
        """ Remove a model from the state machine. The model will still contain all previously added triggers
        and callbacks, but will not receive updates when states or transitions are added to the Machine.
        If an event queue is used, all queued events of that model will be removed."""
        models = listify(model)
        if self.has_queue == 'model':
            for mod in models:
                del self._transition_queue_dict[id(mod)]
                self.models.remove(mod)
        else:
            for mod in models:
                self.models.remove(mod)
        if len(self._transition_queue) > 0:
            queue = self._transition_queue
            new_queue = [queue.popleft()] + [e for e in queue if e.args[0].model not in models]
            self._transition_queue.clear()
            self._transition_queue.extend(new_queue)

    async def _can_trigger(self, model, trigger, *args, **kwargs):
        evt = AsyncEventData(None, None, self, model, args, kwargs)
        state = self.get_model_state(model).name
        for trigger_name in self.get_triggers(state):
            if trigger_name != trigger:
                continue
            for transition in self.events[trigger_name].transitions[state]:
                try:
                    _ = self.get_state(transition.dest)
                except ValueError:
                    continue
                await self.callbacks(self.prepare_event, evt)
                await self.callbacks(transition.prepare, evt)
                if all(await self.await_all([partial(c.check, evt) for c in transition.conditions])):
                    return True
        return False

    def _process(self, trigger):
        raise RuntimeError('AsyncMachine should not call `Machine._process`. Use `Machine._process_async` instead.')

    async def _process_async(self, trigger, model):
        if not self.has_queue:
            if not self._transition_queue:
                return await trigger()
            raise MachineError('Attempt to process events synchronously while transition queue is not empty!')
        self._transition_queue_dict[id(model)].append(trigger)
        if len(self._transition_queue_dict[id(model)]) > 1:
            return True
        while self._transition_queue_dict[id(model)]:
            try:
                await self._transition_queue_dict[id(model)][0]()
            except Exception:
                self._transition_queue_dict[id(model)].clear()
                raise
            try:
                self._transition_queue_dict[id(model)].popleft()
            except KeyError:
                return True
        return True