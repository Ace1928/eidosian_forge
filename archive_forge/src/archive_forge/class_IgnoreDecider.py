import abc
import itertools
from taskflow import deciders
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states
class IgnoreDecider(Decider):
    """Checks any provided edge-deciders and determines if ok to run."""
    _depth_strategies = {deciders.Depth.ALL: _affect_all_successors, deciders.Depth.ATOM: _affect_atom, deciders.Depth.FLOW: _affect_successor_tasks_in_same_flow, deciders.Depth.NEIGHBORS: _affect_direct_task_neighbors}

    def __init__(self, atom, edge_deciders):
        self._atom = atom
        self._edge_deciders = edge_deciders

    def tally(self, runtime):
        voters = {'run_it': [], 'do_not_run_it': [], 'ignored': []}
        history = {}
        if self._edge_deciders:
            states_intentions = runtime.storage.get_atoms_states((ed.from_node.name for ed in self._edge_deciders if ed.kind in compiler.ATOMS))
            for atom_name in states_intentions.keys():
                atom_state, _atom_intention = states_intentions[atom_name]
                if atom_state != states.IGNORE:
                    history[atom_name] = runtime.storage.get(atom_name)
            for ed in self._edge_deciders:
                if ed.kind in compiler.ATOMS and ed.from_node.name not in history:
                    voters['ignored'].append(ed)
                    continue
                if not ed.decider(history=history):
                    voters['do_not_run_it'].append(ed)
                else:
                    voters['run_it'].append(ed)
        if LOG.isEnabledFor(logging.TRACE):
            LOG.trace("Out of %s deciders there were %s 'do no run it' voters, %s 'do run it' voters and %s 'ignored' voters for transition to atom '%s' given history %s", sum((len(eds) for eds in voters.values())), list((ed.from_node.name for ed in voters['do_not_run_it'])), list((ed.from_node.name for ed in voters['run_it'])), list((ed.from_node.name for ed in voters['ignored'])), self._atom.name, history)
        return voters['do_not_run_it']

    def affect(self, runtime, nay_voters):
        widest_depth = deciders.pick_widest((ed.depth for ed in nay_voters))
        affector = self._depth_strategies[widest_depth]
        return affector(self._atom, runtime)