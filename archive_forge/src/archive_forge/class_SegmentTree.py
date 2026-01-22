import operator
from typing import Any, Optional
class SegmentTree:
    """A Segment Tree data structure.

    https://en.wikipedia.org/wiki/Segment_tree

    Can be used as regular array, but with two important differences:

      a) Setting an item's value is slightly slower. It is O(lg capacity),
         instead of O(1).
      b) Offers efficient `reduce` operation which reduces the tree's values
         over some specified contiguous subsequence of items in the array.
         Operation could be e.g. min/max/sum.

    The data is stored in a list, where the length is 2 * capacity.
    The second half of the list stores the actual values for each index, so if
    capacity=8, values are stored at indices 8 to 15. The first half of the
    array contains the reduced-values of the different (binary divided)
    segments, e.g. (capacity=4):
    0=not used
    1=reduced-value over all elements (array indices 4 to 7).
    2=reduced-value over array indices (4 and 5).
    3=reduced-value over array indices (6 and 7).
    4-7: values of the tree.
    NOTE that the values of the tree are accessed by indices starting at 0, so
    `tree[0]` accesses `internal_array[4]` in the above example.
    """

    def __init__(self, capacity: int, operation: Any, neutral_element: Optional[Any]=None):
        """Initializes a Segment Tree object.

        Args:
            capacity: Total size of the array - must be a power of two.
            operation: Lambda obj, obj -> obj
                The operation for combining elements (eg. sum, max).
                Must be a mathematical group together with the set of
                possible values for array elements.
            neutral_element (Optional[obj]): The neutral element for
                `operation`. Use None for automatically finding a value:
                max: float("-inf"), min: float("inf"), sum: 0.0.
        """
        assert capacity > 0 and capacity & capacity - 1 == 0, 'Capacity must be positive and a power of 2!'
        self.capacity = capacity
        if neutral_element is None:
            neutral_element = 0.0 if operation is operator.add else float('-inf') if operation is max else float('inf')
        self.neutral_element = neutral_element
        self.value = [self.neutral_element for _ in range(2 * capacity)]
        self.operation = operation

    def reduce(self, start: int=0, end: Optional[int]=None) -> Any:
        """Applies `self.operation` to subsequence of our values.

        Subsequence is contiguous, includes `start` and excludes `end`.

          self.operation(
              arr[start], operation(arr[start+1], operation(... arr[end])))

        Args:
            start: Start index to apply reduction to.
            end (Optional[int]): End index to apply reduction to (excluded).

        Returns:
            any: The result of reducing self.operation over the specified
                range of `self._value` elements.
        """
        if end is None:
            end = self.capacity
        elif end < 0:
            end += self.capacity
        result = self.neutral_element
        start += self.capacity
        end += self.capacity
        while start < end:
            if start & 1:
                result = self.operation(result, self.value[start])
                start += 1
            if end & 1:
                end -= 1
                result = self.operation(result, self.value[end])
            start //= 2
            end //= 2
        return result

    def __setitem__(self, idx: int, val: float) -> None:
        """
        Inserts/overwrites a value in/into the tree.

        Args:
            idx: The index to insert to. Must be in [0, `self.capacity`[
            val: The value to insert.
        """
        assert 0 <= idx < self.capacity, f'idx={idx} capacity={self.capacity}'
        idx += self.capacity
        self.value[idx] = val
        idx = idx >> 1
        while idx >= 1:
            update_idx = 2 * idx
            self.value[idx] = self.operation(self.value[update_idx], self.value[update_idx + 1])
            idx = idx >> 1

    def __getitem__(self, idx: int) -> Any:
        assert 0 <= idx < self.capacity
        return self.value[idx + self.capacity]

    def get_state(self):
        return self.value

    def set_state(self, state):
        assert len(state) == self.capacity * 2
        self.value = state