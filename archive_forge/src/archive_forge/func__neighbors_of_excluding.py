from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _neighbors_of_excluding(self, qubit: GridQubit, used: Set[GridQubit]) -> List[GridQubit]:
    return [n for n in self._c_adj[qubit] if n not in used]