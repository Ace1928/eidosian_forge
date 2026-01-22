from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _choose_random_edge(self, edges: Set[EDGE]) -> Optional[EDGE]:
    """Picks random edge from the set of edges.

        Args:
          edges: Set of edges to pick from.

        Returns:
          Random edge from the supplied set, or None for empty set.
        """
    if edges:
        index = self._rand.randint(len(edges))
        for e in edges:
            if not index:
                return e
            index -= 1
    return None