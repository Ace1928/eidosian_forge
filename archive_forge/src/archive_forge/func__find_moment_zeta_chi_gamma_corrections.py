import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _find_moment_zeta_chi_gamma_corrections(moment: cirq.Moment, characterization_index: Optional[int], parameters: Optional[PhasedFSimCalibrationResult], gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]]) -> Tuple[List[Tuple[Tuple[cirq.Operation, ...], ...]], Optional[List[Optional[int]]], List[cirq.Operation]]:
    """Finds corrections for each operation within a moment to compensate for zeta, chi and gamma.

    Args:
        moment: Moment to compensate.
        characterization_index: The original characterization index of a moment.
        parameters: Characterizations results for a given moment. None, when not available.
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
            characterization. Defaults to sqrt_iswap_gates_translator.

    Returns:
        Tuple of:
         - decompositions: the decomposed operations for each corrected operation, each element of
           this list is a list of moments of the decomposed gate.
         - decompositions_moment_to_calibration: for each moment in the decomposition, assigns a
           characterization index that matches the original decomposed gate. None when no gate
           was decomposed.
         - other: the remaining gates that were not decomposed.

    Raises:
        IncompatibleMomentError: If a moment has operations different than `cirq.GateOperation`,
            if it contains an unsupported greater than one qubit operation, if parameters are
            missing, if the engine gate does not match the `parameters` gate, or if there is
            missing characterization data for a pair of qubits.
        ValueError: If parameter is not supplied, if the translated engine gate does not match
            the one in parameters, or if pair parameters cannot be obtained.
    """
    default_phases = PhasedFSimCharacterization(zeta=0.0, chi=0.0, gamma=0.0)
    decompositions: List[Tuple[Tuple[cirq.Operation, ...], ...]] = []
    other: List[cirq.Operation] = []
    decompositions_moment_to_calibration: Optional[List[Optional[int]]] = None
    for op in moment:
        if not isinstance(op, cirq.GateOperation):
            raise IncompatibleMomentError('Moment contains operation different than GateOperation')
        if isinstance(op.gate, cirq.GlobalPhaseGate):
            raise IncompatibleMomentError('Moment contains global phase gate')
        if isinstance(op.gate, _CALIBRATION_IRRELEVANT_GATES) or cirq.num_qubits(op.gate) == 1:
            other.append(op)
            continue
        a, b = op.qubits
        translated = gates_translator(op.gate)
        if translated is None:
            raise IncompatibleMomentError(f'Moment {moment} contains unsupported non-single qubit operation {op}')
        if parameters is None:
            raise ValueError(f'Missing characterization data for moment {moment}')
        if translated.engine_gate != parameters.gate:
            raise ValueError(f"Engine gate {translated.engine_gate} doesn't match characterized gate {parameters.gate}")
        pair_parameters = parameters.get_parameters(a, b)
        if pair_parameters is None:
            raise ValueError(f'Missing characterization data for pair {(a, b)} in {parameters}')
        pair_parameters = pair_parameters.merge_with(default_phases)
        corrections = FSimPhaseCorrections.from_characterization((a, b), translated, pair_parameters, characterization_index)
        if decompositions_moment_to_calibration is None:
            decompositions_moment_to_calibration = corrections.moment_to_calibration
        else:
            assert decompositions_moment_to_calibration == corrections.moment_to_calibration, f'Inconsistent decompositions with a moment {moment}'
        decompositions.append(corrections.operations)
    return (decompositions, decompositions_moment_to_calibration, other)