from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _force_edge_active_move(self, state: _STATE) -> _STATE:
    """Move which forces a random edge to appear on some sequence.

        This move chooses random edge from the edges which do not belong to any
        sequence and modifies state in such a way, that this chosen edge
        appears on some sequence of the search state.

        Args:
          state: Search state, not mutated.

        Returns:
          New search state with one of the unused edges appearing in some
          sequence.
        """
    seqs, edges = state
    unused_edges = edges.copy()
    for seq in seqs:
        for i in range(1, len(seq)):
            unused_edges.remove(self._normalize_edge((seq[i - 1], seq[i])))
    edge = self._choose_random_edge(unused_edges)
    if not edge:
        return (seqs, edges)
    return (self._force_edge_active(seqs, edge, lambda: bool(self._rand.randint(2))), edges)