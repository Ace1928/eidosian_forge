from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def _contiguous_groups(length: int, comparator: Callable[[int, int], bool]) -> List[Tuple[int, int]]:
    """Splits range(length) into approximate equivalence classes.

    Args:
        length: The length of the range to split.
        comparator: Determines if two indices have approximately equal items.

    Returns:
        A list of (inclusive_start, exclusive_end) range endpoints. Each
        corresponds to a run of approximately-equivalent items.
    """
    result = []
    start = 0
    while start < length:
        past = start + 1
        while past < length and comparator(start, past):
            past += 1
        result.append((start, past))
        start = past
    return result