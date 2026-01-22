from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
class GreedySequenceSearchStrategy(place_strategy.LinePlacementStrategy):
    """Greedy search method for linear sequence of qubits on a chip."""

    def __init__(self, algorithm: str='best') -> None:
        """Constructs a greedy search strategy object.

        Args:
            algorithm: Greedy algorithm to be used. Available options are:
             -  `best`:  runs all heuristics and chooses the best result,
             -  `largest_area`:  on every step takes the qubit which has
            connection with the largest number of unassigned qubits, and
             -  `minimal_connectivity`: on every step takes the qubit with
            minimal number of unassigned neighbouring qubits.
        """
        self.algorithm = algorithm

    def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
        """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.

        Raises:
            ValueError: If search algorithm passed on initialization is not
                        recognized.
        """
        if not device.metadata.qubit_set:
            return GridQubitLineTuple()
        start: GridQubit = min(device.metadata.qubit_set)
        sequences: List[LineSequence] = []
        greedy_search: Dict[str, List[GreedySequenceSearch]] = {'minimal_connectivity': [_PickFewestNeighbors(device, start)], 'largest_area': [_PickLargestArea(device, start)], 'best': [_PickFewestNeighbors(device, start), _PickLargestArea(device, start)]}
        algos = greedy_search.get(self.algorithm)
        if algos is None:
            raise ValueError(f'Unknown greedy search algorithm {self.algorithm}')
        for algorithm in algos:
            sequences.append(algorithm.get_or_search())
        return GridQubitLineTuple.best_of(sequences, length)