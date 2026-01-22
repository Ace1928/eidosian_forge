from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
class LockedMachine(Machine):
    """ Machine class which manages contexts. In it's default version the machine uses a `threading.Lock`
        context to lock access to its methods and event triggers bound to model objects.
    Attributes:
        machine_context (dict): A dict of context managers to be entered whenever a machine method is
            called or an event is triggered. Contexts are managed for each model individually.
    """
    event_cls = LockedEvent

    def __init__(self, model=Machine.self_literal, states=None, initial='initial', transitions=None, send_event=False, auto_transitions=True, ordered_transitions=False, ignore_invalid_triggers=None, before_state_change=None, after_state_change=None, name=None, queued=False, prepare_event=None, finalize_event=None, model_attribute='state', on_exception=None, machine_context=None, **kwargs):
        self._ident = IdentManager()
        self.machine_context = listify(machine_context) or [PicklableLock()]
        self.machine_context.append(self._ident)
        self.model_context_map = defaultdict(list)
        super(LockedMachine, self).__init__(model=model, states=states, initial=initial, transitions=transitions, send_event=send_event, auto_transitions=auto_transitions, ordered_transitions=ordered_transitions, ignore_invalid_triggers=ignore_invalid_triggers, before_state_change=before_state_change, after_state_change=after_state_change, name=name, queued=queued, prepare_event=prepare_event, finalize_event=finalize_event, model_attribute=model_attribute, on_exception=on_exception, **kwargs)

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items()}
        del state['model_context_map']
        state['_model_context_map_store'] = {mod: self.model_context_map[id(mod)] for mod in self.models}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_context_map = defaultdict(list)
        for model in self.models:
            self.model_context_map[id(model)] = self._model_context_map_store[model]
        del self._model_context_map_store

    def add_model(self, model, initial=None, model_context=None):
        """ Extends `transitions.core.Machine.add_model` by `model_context` keyword.
        Args:
            model (list or object): A model (list) to be managed by the machine.
            initial (str, Enum or State): The initial state of the passed model[s].
            model_context (list or object): If passed, assign the context (list) to the machines
                model specific context map.
        """
        models = listify(model)
        model_context = listify(model_context) if model_context is not None else []
        super(LockedMachine, self).add_model(models, initial)
        for mod in models:
            mod = self if mod is self.self_literal else mod
            self.model_context_map[id(mod)].extend(self.machine_context)
            self.model_context_map[id(mod)].extend(model_context)

    def remove_model(self, model):
        """ Extends `transitions.core.Machine.remove_model` by removing model specific context maps
            from the machine when the model itself is removed. """
        models = listify(model)
        for mod in models:
            del self.model_context_map[id(mod)]
        return super(LockedMachine, self).remove_model(models)

    def __getattribute__(self, item):
        get_attr = super(LockedMachine, self).__getattribute__
        tmp = get_attr(item)
        if not item.startswith('_') and inspect.ismethod(tmp):
            return partial(get_attr('_locked_method'), tmp)
        return tmp

    def __getattr__(self, item):
        try:
            return super(LockedMachine, self).__getattribute__(item)
        except AttributeError:
            return super(LockedMachine, self).__getattr__(item)

    def _add_model_to_state(self, state, model):
        super(LockedMachine, self)._add_model_to_state(state, model)
        for prefix in self.state_cls.dynamic_methods:
            callback = '{0}_{1}'.format(prefix, self._get_qualified_state_name(state))
            func = getattr(model, callback, None)
            if isinstance(func, partial) and func.func != state.add_callback:
                state.add_callback(prefix[3:], callback)

    def _get_qualified_state_name(self, state):
        return state.name

    def _locked_method(self, func, *args, **kwargs):
        if self._ident.current != get_ident():
            with nested(*self.machine_context):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)