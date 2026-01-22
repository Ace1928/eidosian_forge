import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __step3(self) -> int:
    """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
    n = self.n
    count = 0
    for i in range(n):
        for j in range(n):
            if self.marked[i][j] == 1 and (not self.col_covered[j]):
                self.col_covered[j] = True
                count += 1
    if count >= n:
        step = 7
    else:
        step = 4
    return step