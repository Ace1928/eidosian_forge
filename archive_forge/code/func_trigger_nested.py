from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def trigger_nested(self, event_data):
    """ Executes all transitions that match the current state,
        halting as soon as one successfully completes.
        It is up to the machine's configuration of the Event whether processing happens queued (sequentially) or
        whether further Events are processed as they occur. NOTE: This should only
        be called by HierarchicalMachine instances.
        Args:
            event_data (NestedEventData): The currently processed event
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
            self._process(event_data)
            if event_data.result:
                elems = state_path
                while elems:
                    done.add(machine.state_cls.separator.join(elems))
                    elems.pop()
    return event_data.result