import copy
import logging
from time import time
from typing import List, Union, Dict, Callable, Any, Optional, TypeVar
import warnings
from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.providers.models import BackendProperties
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.instruction_durations import InstructionDurations, InstructionDurationsType
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.target import Target, target_to_backend_properties
def _parse_inst_map(inst_map, backend):
    if inst_map is None:
        backend_version = getattr(backend, 'version', 0)
        if backend_version <= 1:
            if hasattr(backend, 'defaults'):
                inst_map = getattr(backend.defaults(), 'instruction_schedule_map', None)
        else:
            inst_map = backend.target.instruction_schedule_map()
    return inst_map