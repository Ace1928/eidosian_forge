import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _extract_all_pairs_to_characterize(circuits: Iterable[cirq.Circuit], gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]], permit_mixed_moments: bool) -> Tuple[Set[Tuple[cirq.Qid, cirq.Qid]], Optional[cirq.Gate]]:
    """Extracts the set of all two-qubit operations from the circuits.

    Args:
        circuits: Circuits to extract the operations from.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.

    Returns:
        Tuple with set of all two-qubit interacting pairs and a common gate that represents those
        interactions. The gate can be used for characterization purposes. If no interactions are
        present the gate is None.

    Raises:
        ValueError: If multiple types of two qubit gates appear in the (possibly translated)
            circuits.
    """
    all_pairs: Set[Tuple[cirq.Qid, cirq.Qid]] = set()
    common_gate = None
    for circuit in circuits:
        for moment in circuit:
            pairs_and_gate = _list_moment_pairs_to_characterize(moment, gates_translator, canonicalize_pairs=True, permit_mixed_moments=permit_mixed_moments, sort_pairs=False)
            if pairs_and_gate is not None:
                pairs, gate = pairs_and_gate
                if common_gate is None:
                    common_gate = gate
                elif common_gate != gate:
                    raise ValueError(f'Only a single type of gate is supported, got {gate} and {common_gate}')
                all_pairs.update(pairs)
    return (all_pairs, common_gate)