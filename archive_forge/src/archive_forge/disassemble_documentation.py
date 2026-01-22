from typing import Any, Dict, List, NewType, Tuple, Union
import collections
import math
from qiskit import pulse
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
Return a list of :class:`qiskit.pulse.Schedule` object(s) from a qobj.

    Args:
        qobj (Qobj): The Qobj object to convert to pulse schedules.

    Returns:
        A list of :class:`qiskit.pulse.Schedule` objects from the qobj

    Raises:
        pulse.PulseError: If a parameterized instruction is supplied.
    