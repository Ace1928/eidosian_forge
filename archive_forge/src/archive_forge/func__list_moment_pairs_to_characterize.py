import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _list_moment_pairs_to_characterize(moment: cirq.Moment, gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]], canonicalize_pairs: bool, permit_mixed_moments: bool, sort_pairs: bool) -> Optional[Tuple[Tuple[Tuple[cirq.Qid, cirq.Qid], ...], cirq.Gate]]:
    """Helper function to describe a given moment in terms of a characterization request.

    Args:
        moment: Moment to characterize.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization.
        canonicalize_pairs: Whether to sort each of the qubit pair so that the first qubit
            is always lower than the second.
        permit_mixed_moments: Whether to allow a mix of two-qubit gates with other irrelevant
            single-qubit gates.
        sort_pairs: Whether to sort all the qubit pairs extracted from the moment which will undergo
            characterization.

    Returns:
        Tuple with list of pairs to characterize and gate that should be used for characterization,
        or None when no gate to characterize exists in a given moment.

    Raises:
        IncompatibleMomentError: When a moment contains operations other than the operations matched
            by gates_translator, or it mixes a single qubit and two qubit gates.
    """
    other_operation = False
    gate: Optional[cirq.FSimGate] = None
    pairs = []
    for op in moment:
        if not isinstance(op, cirq.GateOperation):
            raise IncompatibleMomentError('Moment contains operation different than GateOperation')
        if isinstance(op.gate, cirq.GlobalPhaseGate):
            raise IncompatibleMomentError('Moment contains global phase gate')
        if isinstance(op.gate, _CALIBRATION_IRRELEVANT_GATES) or cirq.num_qubits(op.gate) == 1:
            other_operation = True
        else:
            translated = gates_translator(op.gate)
            if translated is None:
                raise IncompatibleMomentError(f'Moment {moment} contains unsupported non-single qubit operation {op}')
            if gate is not None and gate != translated.engine_gate:
                raise IncompatibleMomentError(f'Moment {moment} contains operations resolved to two different gates {gate} and {translated.engine_gate}')
            else:
                gate = translated.engine_gate
            pair = cast(Tuple[cirq.Qid, cirq.Qid], tuple(sorted(op.qubits) if canonicalize_pairs else op.qubits))
            pairs.append(pair)
    if gate is None:
        return None
    elif not permit_mixed_moments and other_operation:
        raise IncompatibleMomentError('Moment contains mixed two-qubit operations and either single-qubit measurement or wait operations. See permit_mixed_moments option to relax this restriction.')
    if sort_pairs:
        pairs_tuple = tuple(sorted(pairs))
    else:
        pairs_tuple = tuple(pairs)
    return (pairs_tuple, gate)