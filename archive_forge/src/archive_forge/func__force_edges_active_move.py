from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _force_edges_active_move(self, state: _STATE) -> _STATE:
    """Move function which repeats _force_edge_active_move a few times.

        Args:
          state: Search state, not mutated.

        Returns:
          New search state which consists of incremental changes of the
          original state.
        """
    for _ in range(self._rand.randint(1, 4)):
        state = self._force_edge_active_move(state)
    return state