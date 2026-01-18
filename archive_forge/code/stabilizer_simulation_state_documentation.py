import abc
from typing import Any, cast, Generic, Optional, Sequence, TYPE_CHECKING, TypeVar, Union
import numpy as np
import sympy
from cirq import linalg, ops, protocols
from cirq.ops import common_gates, global_phase_op, matrix_gates, swap_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.simulation_state import SimulationState
from cirq.type_workarounds import NotImplementedType
Apply a SWAP gate