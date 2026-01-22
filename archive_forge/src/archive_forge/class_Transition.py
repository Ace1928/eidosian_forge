import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
class Transition(object):
    """ Representation of a transition managed by a ``Machine`` instance.

    Attributes:
        source (str): Source state of the transition.
        dest (str): Destination state of the transition.
        prepare (list): Callbacks executed before conditions checks.
        conditions (list): Callbacks evaluated to determine if
            the transition should be executed.
        before (list): Callbacks executed before the transition is executed
            but only if condition checks have been successful.
        after (list): Callbacks executed after the transition is executed
            but only if condition checks have been successful.
    """
    dynamic_methods = ['before', 'after', 'prepare']
    ' A list of dynamic methods which can be resolved by a ``Machine`` instance for convenience functions. '
    condition_cls = Condition
    " The class used to wrap condition checks. Can be replaced to alter condition resolution behaviour\n        (e.g. OR instead of AND for 'conditions' or AND instead of OR for 'unless') "

    def __init__(self, source, dest, conditions=None, unless=None, before=None, after=None, prepare=None):
        """
        Args:
            source (str): The name of the source State.
            dest (str): The name of the destination State.
            conditions (optional[str, callable or list]): Condition(s) that must pass in order for
                the transition to take place. Either a string providing the
                name of a callable, or a list of callables. For the transition
                to occur, ALL callables must return True.
            unless (optional[str, callable or list]): Condition(s) that must return False in order
                for the transition to occur. Behaves just like conditions arg
                otherwise.
            before (optional[str, callable or list]): callbacks to trigger before the
                transition.
            after (optional[str, callable or list]): callbacks to trigger after the transition.
            prepare (optional[str, callable or list]): callbacks to trigger before conditions are checked
        """
        self.source = source
        self.dest = dest
        self.prepare = [] if prepare is None else listify(prepare)
        self.before = [] if before is None else listify(before)
        self.after = [] if after is None else listify(after)
        self.conditions = []
        if conditions is not None:
            for cond in listify(conditions):
                self.conditions.append(self.condition_cls(cond))
        if unless is not None:
            for cond in listify(unless):
                self.conditions.append(self.condition_cls(cond, target=False))

    def _eval_conditions(self, event_data):
        for cond in self.conditions:
            if not cond.check(event_data):
                _LOGGER.debug('%sTransition condition failed: %s() does not return %s. Transition halted.', event_data.machine.name, cond.func, cond.target)
                return False
        return True

    def execute(self, event_data):
        """ Execute the transition.
        Args:
            event_data: An instance of class EventData.
        Returns: boolean indicating whether the transition was
            successfully executed (True if successful, False if not).
        """
        _LOGGER.debug('%sInitiating transition from state %s to state %s...', event_data.machine.name, self.source, self.dest)
        event_data.machine.callbacks(self.prepare, event_data)
        _LOGGER.debug('%sExecuted callbacks before conditions.', event_data.machine.name)
        if not self._eval_conditions(event_data):
            return False
        event_data.machine.callbacks(itertools.chain(event_data.machine.before_state_change, self.before), event_data)
        _LOGGER.debug('%sExecuted callback before transition.', event_data.machine.name)
        if self.dest:
            self._change_state(event_data)
        event_data.machine.callbacks(itertools.chain(self.after, event_data.machine.after_state_change), event_data)
        _LOGGER.debug('%sExecuted callback after transition.', event_data.machine.name)
        return True

    def _change_state(self, event_data):
        event_data.machine.get_state(self.source).exit(event_data)
        event_data.machine.set_state(self.dest, event_data.model)
        event_data.update(getattr(event_data.model, event_data.machine.model_attribute))
        event_data.machine.get_state(self.dest).enter(event_data)

    def add_callback(self, trigger, func):
        """ Add a new before, after, or prepare callback.
        Args:
            trigger (str): The type of triggering event. Must be one of
                'before', 'after' or 'prepare'.
            func (str or callable): The name of the callback function or a callable.
        """
        callback_list = getattr(self, trigger)
        callback_list.append(func)

    def __repr__(self):
        return "<%s('%s', '%s')@%s>" % (type(self).__name__, self.source, self.dest, id(self))