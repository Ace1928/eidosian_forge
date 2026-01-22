import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
@abc.abstractmethod
def _op_symbol(self) -> str:
    """The symbol used when visualizing the composite operator"""