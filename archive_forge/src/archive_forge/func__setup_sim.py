from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals
def _setup_sim(self) -> None:
    if _optionals.HAS_AER:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        self._sim = AerSimulator()
        noise_model = NoiseModel.from_backend(self)
        self._sim.set_options(noise_model=noise_model)
        self.set_options(noise_model=noise_model)
    else:
        self._sim = BasicSimulator()