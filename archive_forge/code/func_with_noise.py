import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def with_noise(self, noise: 'cirq.NOISE_MODEL_LIKE') -> 'cirq.Circuit':
    """Make a noisy version of the circuit.

        Args:
            noise: The noise model to use.  This describes the kind of noise to
                add to the circuit.

        Returns:
            A new circuit with the same moment structure but with new moments
            inserted where needed when more than one noisy operation is
            generated for an input operation.  Emptied moments are removed.
        """
    noise_model = devices.NoiseModel.from_noise_model_like(noise)
    qubits = sorted(self.all_qubits())
    c_noisy = Circuit()
    for op_tree in noise_model.noisy_moments(self, qubits):
        c_noisy += Circuit(op_tree)
    return c_noisy