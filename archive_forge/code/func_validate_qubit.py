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
def validate_qubit(self, qubit: 'cirq.Qid') -> None:
    """Raises an exception if the qubit does not satisfy the topological constraints
        of the RigettiQCSAspenDevice.

        Args:
            qubit: The qubit to validate.

        Raises:
            UnsupportedQubit: The operation isn't valid for this device.
        """
    if isinstance(qubit, cirq.GridQubit):
        if self._number_octagons < 2:
            raise UnsupportedQubit('this device does not support GridQubits')
        if not (qubit.row <= 1 and qubit.col <= 1):
            raise UnsupportedQubit('Aspen devices only support square grids of 1 row and 1 column')
        return
    if isinstance(qubit, cirq.LineQubit):
        if not qubit.x <= self._number_octagons * 8:
            raise UnsupportedQubit('this Aspen device only supports line ', f'qubits up to length {self._number_octagons * 8}')
        return
    if isinstance(qubit, cirq.NamedQubit):
        try:
            index = int(qubit.name)
            if not index < self._maximum_qubit_number:
                raise UnsupportedQubit(f'this Aspen device only supports qubits up to index {self._maximum_qubit_number}')
            if not index % 10 <= 7:
                raise UnsupportedQubit('this Aspen device only supports qubit indices mod 10 <= 7')
            return
        except ValueError:
            raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')
    if isinstance(qubit, (OctagonalQubit, AspenQubit)):
        if not qubit.index < self._maximum_qubit_number:
            raise UnsupportedQubit('this Aspen device only supports ', f'qubits up to index {self._maximum_qubit_number}')
        return
    else:
        raise UnsupportedQubit(f'unsupported Qid type {type(qubit)}')