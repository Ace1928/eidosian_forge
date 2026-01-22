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
@transpile.custom_qnode_transform
def _transpile_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``transpile``."""
    if tkwargs.get('device', None):
        raise ValueError("Cannot provide a 'device' value directly to the defer_measurements decorator when transforming a QNode.")
    tkwargs.setdefault('device', qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)