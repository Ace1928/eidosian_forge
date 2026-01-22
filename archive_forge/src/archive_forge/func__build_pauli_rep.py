import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@abc.abstractmethod
def _build_pauli_rep(self):
    """The function to generate the pauli representation for the composite operator."""