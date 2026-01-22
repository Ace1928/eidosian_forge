import operator
from typing import Any, Optional
class SumSegmentTree(SegmentTree):
    """A SegmentTree with the reduction `operation`=operator.add."""

    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add)

    def sum(self, start: int=0, end: Optional[Any]=None) -> Any:
        """Returns the sum over a sub-segment of the tree."""
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Finds highest i, for which: sum(arr[0]+..+arr[i - i]) <= prefixsum.

        Args:
            prefixsum: `prefixsum` upper bound in above constraint.

        Returns:
            int: Largest possible index (i) satisfying above constraint.
        """
        assert 0 <= prefixsum <= self.sum() + 1e-05
        idx = 1
        while idx < self.capacity:
            update_idx = 2 * idx
            if self.value[update_idx] > prefixsum:
                idx = update_idx
            else:
                prefixsum -= self.value[update_idx]
                idx = update_idx + 1
        return idx - self.capacity