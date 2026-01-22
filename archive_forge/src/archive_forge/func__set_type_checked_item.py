from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
def _set_type_checked_item(self, virtual, physical):
    old = self._v2p.pop(virtual, None)
    self._p2v.pop(old, None)
    old = self._p2v.pop(physical, None)
    self._v2p.pop(old, None)
    self._p2v[physical] = virtual
    if virtual is not None:
        self._v2p[virtual] = physical