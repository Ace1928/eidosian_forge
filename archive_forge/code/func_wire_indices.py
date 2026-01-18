from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def wire_indices(self, wire):
    """Operator indices on the given wire.

        Args:
            wire (int): wire to examine

        Returns:
            list[int]: indices of operators on the wire, in temporal order
        """
    return [op.queue_idx for op in self._grid[wire]]