from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _find_sequence(self) -> List[GridQubit]:
    """Looks for a sequence starting at a given qubit.

        Search is issued twice from the starting qubit, so that longest possible
        sequence is found. Starting qubit might not be the first qubit on the
        returned sequence.

        Returns:
            The longest sequence found by this method.
        """
    tail = self._sequence_search(self._start, [])
    tail.pop(0)
    head = self._sequence_search(self._start, tail)
    head.reverse()
    return self._expand_sequence(head + tail)