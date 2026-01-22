from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
def apply_swap(self, lq1: int, lq2: int) -> None:
    """Updates the mapping to simulate inserting a swap operation between `lq1` and `lq2`.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Raises:
            ValueError: whenever lq1 and lq2 are not adjacent on the device.
        """
    if self.dist_on_device(lq1, lq2) > 1:
        raise ValueError(f'q1: {lq1} and q2: {lq2} are not adjacent on the device. Cannot swap them.')
    pq1, pq2 = (self.logical_to_physical[lq1], self.logical_to_physical[lq2])
    self._logical_to_physical[[lq1, lq2]] = self._logical_to_physical[[lq2, lq1]]
    self._physical_to_logical[[pq1, pq2]] = self._physical_to_logical[[pq2, pq1]]