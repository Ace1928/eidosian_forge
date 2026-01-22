import dataclasses
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import cirq
from cirq.experiments import HALF_GRID_STAGGERED_PATTERN
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
from cirq_google.calibration.xeb_wrapper import run_local_xeb_calibration
from cirq_google.engine import AbstractProcessor, AbstractEngine, ProcessorSampler
def _make_zeta_chi_gamma_compensation(circuit_with_calibration: CircuitWithCalibration, characterizations: List[PhasedFSimCalibrationResult], gates_translator: Callable[[cirq.Gate], Optional[PhaseCalibratedFSimGate]], permit_mixed_moments: bool) -> CircuitWithCalibration:
    if len(circuit_with_calibration.circuit) != len(circuit_with_calibration.moment_to_calibration):
        raise ValueError('Moment allocations does not match circuit length')
    compensated = cirq.Circuit()
    compensated_moment_to_calibration: List[Optional[int]] = []
    for moment, characterization_index in zip(circuit_with_calibration.circuit, circuit_with_calibration.moment_to_calibration):
        parameters = None
        if characterization_index is not None:
            parameters = characterizations[characterization_index]
        decompositions, decompositions_moment_to_calibration, other = _find_moment_zeta_chi_gamma_corrections(moment, characterization_index, parameters, gates_translator)
        if decompositions:
            assert decompositions_moment_to_calibration is not None
            if not other:
                moment_to_calibration_index: Optional[int] = None
            else:
                if not permit_mixed_moments:
                    raise IncompatibleMomentError(f'Moment {moment} contains mixed operations. See permit_mixed_moments option to relax this restriction.')
                moment_to_calibration_index, = [index for index, moment_to_calibration in enumerate(decompositions_moment_to_calibration) if moment_to_calibration is not None]
            for index, operations in enumerate(itertools.zip_longest(*decompositions, fillvalue=())):
                compensated += cirq.Moment(operations, other if index == moment_to_calibration_index else ())
            compensated_moment_to_calibration += decompositions_moment_to_calibration
        elif other:
            compensated += cirq.Moment(other)
            compensated_moment_to_calibration.append(characterization_index)
    return CircuitWithCalibration(compensated, compensated_moment_to_calibration)