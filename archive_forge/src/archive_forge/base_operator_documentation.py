from __future__ import annotations
import copy
from abc import ABC
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from .mixins import GroupMixin
Make a deep copy of current operator.