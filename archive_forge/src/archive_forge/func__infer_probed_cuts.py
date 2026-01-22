import warnings
from collections.abc import Sequence as SequenceType
from dataclasses import InitVar, dataclass
from typing import Any, ClassVar, Dict, List, Sequence, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.ops.meta import WireCut
def _infer_probed_cuts(self, wire_depths, max_wires_by_fragment=None, max_gates_by_fragment=None, exhaustive=True) -> List[Dict[str, Any]]:
    """
        Helper function for deriving the minimal set of best default partitioning constraints
        for the graph partitioner.

        Args:
            num_tape_wires (int): Number of wires in the circuit tape to be partitioned.
            num_tape_gates (int): Number of gates in the circuit tape to be partitioned.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                ``num_tape_gates - 1`` attempts will be made with ``num_fragments`` ranging from
                [2, ``num_tape_gates``], i.e. from bi-partitioning to complete partitioning where
                each fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal set of kwargs being passed to a graph
                partitioner method.
        """
    num_tape_wires = len(wire_depths)
    num_tape_gates = int(sum(wire_depths.values()))
    max_free_wires = self.max_free_wires or num_tape_wires
    max_free_gates = self.max_free_gates or num_tape_gates
    min_free_wires = self.min_free_wires or max_free_wires
    min_free_gates = self.min_free_gates or max_free_gates
    k_lb = 1 + max((num_tape_wires - 1) // max_free_wires, (num_tape_gates - 1) // max_free_gates)
    k_ub = 1 + max((num_tape_wires - 1) // min_free_wires, (num_tape_gates - 1) // min_free_gates)
    if exhaustive:
        k_lb = max(2, k_lb)
        k_ub = num_tape_gates
    imbalance_tolerance = k_ub if self.imbalance_tolerance is None else self.imbalance_tolerance
    probed_cuts = []
    if max_gates_by_fragment is None and max_wires_by_fragment is None:
        k_lower = self.k_lower if self.k_lower is not None else k_lb
        k_upper = self.k_upper if self.k_upper is not None else k_ub
        if k_lower < k_lb:
            warnings.warn(f'The provided `k_lower={k_lower}` is less than the lowest allowed value, will override and set `k_lower={k_lb}`.')
            k_lower = k_lb
        if k_lower > self.HIGH_NUM_FRAGMENTS:
            warnings.warn(f'The attempted number of fragments seems high with lower bound at {k_lower}.')
        ks = list(range(k_lower, k_upper + 1))
        if len(ks) > self.HIGH_PARTITION_ATTEMPTS:
            warnings.warn(f'The numer of partition attempts seems high ({len(ks)}).')
    else:
        ks = [len(max_wires_by_fragment or max_gates_by_fragment)]
    for k in ks:
        imbalance = self._infer_imbalance(k, wire_depths, max_free_wires if max_wires_by_fragment is None else max(max_wires_by_fragment), max_free_gates if max_gates_by_fragment is None else max(max_gates_by_fragment), imbalance_tolerance)
        cut_kwargs = {'num_fragments': k, 'imbalance': imbalance}
        if max_wires_by_fragment is not None:
            cut_kwargs['max_wires_by_fragment'] = max_wires_by_fragment
        if max_gates_by_fragment is not None:
            cut_kwargs['max_gates_by_fragment'] = max_gates_by_fragment
        probed_cuts.append(cut_kwargs)
    return probed_cuts