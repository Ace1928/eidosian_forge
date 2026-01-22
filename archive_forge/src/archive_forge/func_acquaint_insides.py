import functools
import itertools
import math
import operator
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, protocols, value
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
def acquaint_insides(swap_gate: 'cirq.Gate', acquaintance_gate: 'cirq.Operation', qubits: Sequence['cirq.Qid'], before: bool, layers: Layers, mapping: Dict[ops.Qid, int]) -> None:
    """Acquaints each of the qubits with another set specified by an
    acquaintance gate.

    Args:
        qubits: The list of qubits of which half are individually acquainted
            with another list of qubits.
        layers: The layers to put gates into.
        acquaintance_gate: The acquaintance gate that acquaints the end qubit
            with another list of qubits.
        before: Whether the acquainting is done before the shift.
        swap_gate: The gate used to swap logical indices.
        mapping: The mapping from qubits to logical indices. Used to keep track
            of the effect of inside-acquainting swaps.
    """
    max_reach = _get_max_reach(len(qubits), round_up=before)
    reaches = itertools.chain(range(1, max_reach + 1), range(max_reach, -1, -1))
    offsets = (0, 1) * max_reach
    swap_gate = SwapPermutationGate(swap_gate)
    ops = []
    for offset, reach in zip(offsets, reaches):
        if offset == before:
            ops.append(acquaintance_gate)
        for dr in range(offset, reach, 2):
            ops.append(swap_gate(*qubits[dr:dr + 2]))
    intrastitial_layer = getattr(layers, 'pre' if before else 'post')
    intrastitial_layer += ops
    interstitial_layer = getattr(layers, ('prior' if before else 'posterior') + '_interstitial')
    interstitial_layer.append(acquaintance_gate)
    reached_qubits = qubits[:max_reach + 1]
    positions = list((mapping[q] for q in reached_qubits))
    mapping.update(zip(reached_qubits, reversed(positions)))