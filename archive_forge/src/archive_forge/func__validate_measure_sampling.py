from __future__ import annotations
import uuid
import time
import logging
import warnings
from collections import Counter
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import Provider
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.options import Options
from qiskit.qobj import QasmQobj, QasmQobjConfig, QasmQobjExperiment
from qiskit.result import Result
from qiskit.transpiler import Target
from .basic_provider_job import BasicProviderJob
from .basic_provider_tools import single_gate_matrix
from .basic_provider_tools import SINGLE_QUBIT_GATES
from .basic_provider_tools import cx_gate_matrix
from .basic_provider_tools import einsum_vecmul_index
from .exceptions import BasicProviderError
def _validate_measure_sampling(self, experiment: QasmQobjExperiment) -> None:
    """Determine if measure sampling is allowed for an experiment

        Args:
            experiment: a qobj experiment.
        """
    if self._shots <= 1:
        self._sample_measure = False
        return
    if hasattr(experiment.config, 'allows_measure_sampling'):
        self._sample_measure = experiment.config.allows_measure_sampling
    else:
        measure_flag = False
        for instruction in experiment.instructions:
            if instruction.name == 'reset':
                self._sample_measure = False
                return
            if measure_flag:
                if instruction.name not in ['measure', 'barrier', 'id', 'u0']:
                    self._sample_measure = False
                    return
            elif instruction.name == 'measure':
                measure_flag = True
        self._sample_measure = True