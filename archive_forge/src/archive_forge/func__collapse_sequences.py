import difflib
from bisect import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple
def _collapse_sequences(matches):
    """Find sequences of lines.

    Given a sequence of [(line_in_a, line_in_b),]
    find regions where they both increment at the same time
    """
    answer = []
    start_a = start_b = None
    length = 0
    for i_a, i_b in matches:
        if start_a is not None and i_a == start_a + length and (i_b == start_b + length):
            length += 1
        else:
            if start_a is not None:
                answer.append((start_a, start_b, length))
            start_a = i_a
            start_b = i_b
            length = 1
    if length != 0:
        answer.append((start_a, start_b, length))
    return answer