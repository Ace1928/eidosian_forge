from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
def to_grid_qubit(self) -> cirq.GridQubit:
    """Converts `AspenQubit` to `cirq.GridQubit`.

        Returns:
            The equivalent GridQubit.

        Raises:
            ValueError: AspenQubit cannot be converted to GridQubit.
        """
    for grid_qubit, aspen_index in _grid_qubit_mapping.items():
        if self.index == aspen_index:
            return grid_qubit
    raise ValueError(f'cannot use {self} as a GridQubit')