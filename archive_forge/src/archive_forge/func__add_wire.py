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
def _add_wire(self, wire):
    """Add a qubit or bit to the circuit.

        Args:
            wire (Bit): the wire to be added

            This adds a pair of in and out nodes connected by an edge.

        Raises:
            DAGCircuitError: if trying to add duplicate wire
        """
    if wire not in self._wires:
        self._wires.add(wire)
        inp_node = DAGInNode(wire=wire)
        outp_node = DAGOutNode(wire=wire)
        input_map_id, output_map_id = self._multi_graph.add_nodes_from([inp_node, outp_node])
        inp_node._node_id = input_map_id
        outp_node._node_id = output_map_id
        self.input_map[wire] = inp_node
        self.output_map[wire] = outp_node
        self._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, wire)
    else:
        raise DAGCircuitError(f'duplicate wire {wire}')