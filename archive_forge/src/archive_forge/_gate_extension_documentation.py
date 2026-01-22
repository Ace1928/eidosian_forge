from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, CYGate, CZGate
from qiskit.circuit.library.standard_gates import TGate, TdgGate, SGate, SdgGate, RZGate, U1Gate
from qiskit.circuit.library.standard_gates import SwapGate, CSwapGate, CRZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals

Dynamically extend Gate classes with functions required for the Hoare
optimizer, namely triviality-conditions and post-conditions.
If `_trivial_if` returns `True` and the qubit is in a classical state
then the gate is trivial.
If a gate has no `_trivial_if`, then is assumed to be non-trivial.
If a gate has no `_postconditions`, then is assumed to have unknown post-conditions.
