import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def _get_maybe_ready_for_execute(self, atom):
    """Returns if an atom is *likely* ready to be executed."""

    def ready_checker(pred_connected_it):
        for pred in pred_connected_it:
            pred_atom, (pred_atom_state, pred_atom_intention) = pred
            if pred_atom_state in (st.SUCCESS, st.IGNORE) and pred_atom_intention in (st.EXECUTE, st.IGNORE):
                continue
            LOG.trace("Unable to begin to execute since predecessor atom '%s' is in state %s with intention %s", pred_atom, pred_atom_state, pred_atom_intention)
            return False
        LOG.trace("Able to let '%s' execute", atom)
        return True
    decider_fetcher = lambda: deciders.IgnoreDecider(atom, self._runtime.fetch_edge_deciders(atom))
    connected_fetcher = lambda: traversal.depth_first_iterate(self._execution_graph, atom, traversal.Direction.BACKWARD)
    LOG.trace("Checking if '%s' is ready to execute", atom)
    return self._get_maybe_ready(atom, st.RUNNING, [st.EXECUTE], connected_fetcher, ready_checker, decider_fetcher, for_what='execute')