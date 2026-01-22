import copy
import logging
import uuid
import warnings
from time import time
from typing import Dict, List, Optional, Union
import numpy as np
from qiskit.assembler import assemble_circuits, assemble_schedules
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import Parameter, QuantumCircuit, Qubit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import Instruction, LoConfig, Schedule, ScheduleBlock
from qiskit.pulse.channels import PulseChannel
from qiskit.qobj import QasmQobj, PulseQobj, QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
def _check_lo_freqs(lo_freq: Union[List[float], None], lo_range: Union[List[float], None], lo_type: str):
    """Check that LO frequencies are within the perscribed LO range.

    NOTE: Only checks if frequency/range lists have equal length. And does not check that the lists
    have length ``n_qubits``. This is because some backends, like simulator backends, do not
    require these constraints. For real hardware, these parameters will be validated on the backend.

    Args:
        lo_freq: List of LO frequencies.
        lo_range: Nested list of LO frequency ranges. Inner list is of the form
            ``[lo_min, lo_max]``.
        lo_type: The type of LO value--"qubit" or "meas".

    Raises:
        QiskitError:
            - If each element of the LO range is not a 2d list.
            - If the LO frequency is not in the LO range for a given qubit.
    """
    if lo_freq and lo_range and (len(lo_freq) == len(lo_range)):
        for i, freq in enumerate(lo_freq):
            freq_range = lo_range[i]
            if not (isinstance(freq_range, list) and len(freq_range) == 2):
                raise QiskitError(f'Each element of {lo_type} LO range must be a 2d list.')
            if freq < freq_range[0] or freq > freq_range[1]:
                raise QiskitError('Qubit {} {} LO frequency is {}. The range is [{}, {}].'.format(i, lo_type, freq, freq_range[0], freq_range[1]))