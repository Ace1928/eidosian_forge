from functools import partial
from typing import List, Union, Sequence, Callable
import networkx as nx
import pennylane as qml
from pennylane.transforms import transform
from pennylane import Hamiltonian
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.ops.qubit import SWAP
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape
A postprocesing function returned by a transform that only converts the batch of results
            into a result for a single ``QuantumTape``.
            