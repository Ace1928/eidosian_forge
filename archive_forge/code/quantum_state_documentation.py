from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
Marginalize a probability vector according to subsystems.

        Args:
            probs (np.array): a probability vector Numpy array.
            dims (tuple): subsystem dimensions.
            qargs (None or list): a list of subsystems to return
                marginalized probabilities for. If None return all
                probabilities (Default: None).

        Returns:
            np.array: the marginalized probability vector flattened
                      for the specified qargs.
        