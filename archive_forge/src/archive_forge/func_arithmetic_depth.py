import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
def arithmetic_depth(self) -> int:
    return 1 + max((op.arithmetic_depth for op in self))