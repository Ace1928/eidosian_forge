import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _prune_phis(self, runner):
    _logger.debug('Prune PHIs'.center(60, '-'))

    def get_used_phis_per_state():
        used_phis = defaultdict(set)
        phi_set = set()
        for state in runner.finished:
            used = set(state._used_regs)
            phis = set(state._phis)
            used_phis[state] |= phis & used
            phi_set |= phis
        return (used_phis, phi_set)

    def find_use_defs():
        defmap = {}
        phismap = defaultdict(set)
        for state in runner.finished:
            for phi, rhs in state._outgoing_phis.items():
                if rhs not in phi_set:
                    defmap[phi] = state
                phismap[phi].add((rhs, state))
        _logger.debug('defmap: %s', _lazy_pformat(defmap))
        _logger.debug('phismap: %s', _lazy_pformat(phismap))
        return (defmap, phismap)

    def propagate_phi_map(phismap):
        """An iterative dataflow algorithm to find the definition
            (the source) of each PHI node.
            """
        blacklist = defaultdict(set)
        while True:
            changing = False
            for phi, defsites in sorted(list(phismap.items())):
                for rhs, state in sorted(list(defsites)):
                    if rhs in phi_set:
                        defsites |= phismap[rhs]
                        blacklist[phi].add((rhs, state))
                to_remove = blacklist[phi]
                if to_remove & defsites:
                    defsites -= to_remove
                    changing = True
            _logger.debug('changing phismap: %s', _lazy_pformat(phismap))
            if not changing:
                break

    def apply_changes(used_phis, phismap):
        keep = {}
        for state, used_set in used_phis.items():
            for phi in used_set:
                keep[phi] = phismap[phi]
        _logger.debug('keep phismap: %s', _lazy_pformat(keep))
        new_out = defaultdict(dict)
        for phi in keep:
            for rhs, state in keep[phi]:
                new_out[state][phi] = rhs
        _logger.debug('new_out: %s', _lazy_pformat(new_out))
        for state in runner.finished:
            state._outgoing_phis.clear()
            state._outgoing_phis.update(new_out[state])
    used_phis, phi_set = get_used_phis_per_state()
    _logger.debug('Used_phis: %s', _lazy_pformat(used_phis))
    defmap, phismap = find_use_defs()
    propagate_phi_map(phismap)
    apply_changes(used_phis, phismap)
    _logger.debug('DONE Prune PHIs'.center(60, '-'))