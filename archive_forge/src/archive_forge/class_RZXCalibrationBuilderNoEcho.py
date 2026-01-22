from __future__ import annotations
import enum
import warnings
from collections.abc import Sequence
from math import pi, erf
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable
class RZXCalibrationBuilderNoEcho(RZXCalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate.

    The ``RZXCalibrationBuilderNoEcho`` is a variation of the
    :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder` pass
    that creates calibrations for the cross-resonance pulses without inserting
    the echo pulses in the pulse schedule. This enables exposing the echo in
    the cross-resonance sequence as gates so that the transpiler can simplify them.
    The ``RZXCalibrationBuilderNoEcho`` only supports the hardware-native direction
    of the CX gate.
    """

    def get_calibration(self, node_op: CircuitInst, qubits: list) -> Schedule | ScheduleBlock:
        """Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified,
                or the backend does not natively support the specified direction of the cx.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError('Target rotation angle is not assigned.') from ex
        if np.isclose(theta, 0.0):
            return ScheduleBlock(name='rzx(0.000)')
        cal_type, cr_tones, comp_tones = _check_calibration_type(self._inst_map, qubits)
        if cal_type in [CRCalType.DIRECT_CX_FORWARD, CRCalType.DIRECT_CX_REVERSE]:
            if self._verbose:
                warnings.warn(f'CR instruction for qubits {qubits} is likely {cal_type.value} sequence. Pulse stretch for this calibration is not currently implemented. RZX schedule is not generated for this qubit pair.', UserWarning)
            raise CalibrationNotAvailable
        if cal_type in [CRCalType.ECR_CX_FORWARD, CRCalType.ECR_FORWARD]:
            with builder.build(default_alignment='left', name='rzx(%.3f)' % theta) as rzx_theta:
                stretched_dur = self.rescale_cr_inst(cr_tones[0], 2 * theta)
                self.rescale_cr_inst(comp_tones[0], 2 * theta)
                builder.delay(stretched_dur, DriveChannel(qubits[0]))
            return rzx_theta
        raise QiskitError('RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates.')