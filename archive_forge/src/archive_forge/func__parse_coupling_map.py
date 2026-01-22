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
def _parse_coupling_map(coupling_map, backend):
    if coupling_map is None:
        backend_version = getattr(backend, 'version', 0)
        if backend_version <= 1:
            if getattr(backend, 'configuration', None):
                configuration = backend.configuration()
                if hasattr(configuration, 'coupling_map') and configuration.coupling_map:
                    coupling_map = CouplingMap(configuration.coupling_map)
        else:
            coupling_map = backend.coupling_map
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        return coupling_map
    if isinstance(coupling_map, list) and all((isinstance(i, list) and len(i) == 2 for i in coupling_map)):
        return CouplingMap(coupling_map)
    else:
        raise TranspilerError('Only a single input coupling map can be used with transpile() if you need to target different coupling maps for different circuits you must call transpile() multiple times')