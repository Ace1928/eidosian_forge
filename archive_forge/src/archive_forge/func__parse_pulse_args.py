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
def _parse_pulse_args(backend, meas_level, meas_return, meas_map, memory_slot_size, rep_time, parametric_pulses, **run_config):
    """Build a pulse RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    Raises:
        QiskitError: If the given meas_level is not allowed for the given `backend`.
    """
    backend_config = None
    if backend:
        backend_config = backend.configuration()
        if meas_level not in getattr(backend_config, 'meas_levels', [MeasLevel.CLASSIFIED]):
            raise QiskitError('meas_level = {} not supported for backend {}, only {} is supported'.format(meas_level, backend_config.backend_name, backend_config.meas_levels))
    meas_map = meas_map or getattr(backend_config, 'meas_map', None)
    dynamic_reprate_enabled = getattr(backend_config, 'dynamic_reprate_enabled', False)
    rep_time = rep_time or getattr(backend_config, 'rep_times', None)
    if rep_time:
        if dynamic_reprate_enabled:
            warnings.warn("Dynamic rep rates are supported on this backend. 'rep_delay' will be used instead of 'rep_time'.", RuntimeWarning)
        if isinstance(rep_time, list):
            rep_time = rep_time[0]
        rep_time = int(rep_time * 1000000.0)
    if parametric_pulses is None:
        parametric_pulses = getattr(backend_config, 'parametric_pulses', [])
    run_config_dict = dict(meas_level=meas_level, meas_return=meas_return, meas_map=meas_map, memory_slot_size=memory_slot_size, rep_time=rep_time, parametric_pulses=parametric_pulses, **run_config)
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})
    return run_config