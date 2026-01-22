import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def _get_maybe_ready(self, atom, transition_to, allowed_intentions, connected_fetcher, ready_checker, decider_fetcher, for_what='?'):

    def iter_connected_states():
        for atom in connected_fetcher():
            atom_states = self._storage.get_atoms_states([atom.name])
            yield (atom, atom_states[atom.name])
    state = self._storage.get_atom_state(atom.name)
    ok_to_transition = self._runtime.check_atom_transition(atom, state, transition_to)
    if not ok_to_transition:
        LOG.trace("Atom '%s' is not ready to %s since it can not transition to %s from its current state %s", atom, for_what, transition_to, state)
        return (False, None)
    intention = self._storage.get_atom_intention(atom.name)
    if intention not in allowed_intentions:
        LOG.trace("Atom '%s' is not ready to %s since its current intention %s is not in allowed intentions %s", atom, for_what, intention, allowed_intentions)
        return (False, None)
    ok_to_run = ready_checker(iter_connected_states())
    if not ok_to_run:
        return (False, None)
    else:
        return (True, decider_fetcher())