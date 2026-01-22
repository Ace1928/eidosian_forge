import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
@abc.abstractmethod
def _math_op(self) -> Callable:
    """The function used when combining the operands of the composite operator"""