from node A to node B means that the (qu)bit passes from the output of A
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
def _is_wire_idle(self, wire):
    """Check if a wire is idle.

        Args:
            wire (Bit): a wire in the circuit.

        Returns:
            bool: true if the wire is idle, false otherwise.

        Raises:
            DAGCircuitError: the wire is not in the circuit.
        """
    if wire not in self._wires:
        raise DAGCircuitError('wire %s not in circuit' % wire)
    try:
        child = next(self.successors(self.input_map[wire]))
    except StopIteration as e:
        raise DAGCircuitError('Invalid dagcircuit input node %s has no output' % self.input_map[wire]) from e
    return child is self.output_map[wire]