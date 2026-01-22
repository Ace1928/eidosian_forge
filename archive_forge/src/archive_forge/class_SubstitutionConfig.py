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
class SubstitutionConfig:
    """Class to store the configuration of a given match substitution, which circuit gates, template gates,
    qubits and predecessors of the match in the circuit.
    """

    def __init__(self, circuit_config, template_config, pred_block, qubit_config, template_dag):
        self.template_dag = template_dag
        self.circuit_config = circuit_config
        self.template_config = template_config
        self.qubit_config = qubit_config
        self.pred_block = pred_block