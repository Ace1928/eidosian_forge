import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
def _update_circuit_info(self):
    """Update circuit metadata

        Sets:
            wires (~.Wires): Wires
            num_wires (int): Number of wires
        """
    self.wires = Wires.all_wires(dict.fromkeys((op.wires for op in self)))
    self.num_wires = len(self.wires)