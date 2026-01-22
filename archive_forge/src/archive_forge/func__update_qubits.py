import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def _update_qubits(circuit_dag, qubits_conf):
    """
    Update the qubits, target qubits and the control qubits given the mapping configuration between the circuit
    and the pattern.

    Args:
        circuit_dag (.CommutationDAG): the DAG representation of the circuit.
        qubits_conf (list): list of qubits of the mapping configuration.
    Return:
        list(list(int)): Wires
        list(list(int)): Target wires
        list(list(int)): Control wires
    """
    wires = []
    control_wires = []
    target_wires = []
    for i, node in enumerate(circuit_dag.get_nodes()):
        wires.append([])
        for q in node[1].wires:
            if q in qubits_conf:
                wires[i].append(qubits_conf.index(q))
        if len(node[1].wires) != len(wires[i]):
            wires[i] = []
        control_wires.append([])
        for q in node[1].control_wires:
            if q in qubits_conf:
                control_wires[i].append(qubits_conf.index(q))
        if len(node[1].control_wires) != len(control_wires[i]):
            control_wires[i] = []
        target_wires.append([])
        for q in node[1].target_wires:
            if q in qubits_conf:
                target_wires[i].append(qubits_conf.index(q))
        if len(node[1].target_wires) != len(target_wires[i]):
            target_wires[i] = []
    return (wires, target_wires, control_wires)