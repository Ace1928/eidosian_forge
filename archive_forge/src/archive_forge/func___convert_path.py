import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable
def __convert_path(self, path: Sequence[Sequence[int]], count: int) -> None:
    for i in range(count + 1):
        if self.marked[path[i][0]][path[i][1]] == 1:
            self.marked[path[i][0]][path[i][1]] = 0
        else:
            self.marked[path[i][0]][path[i][1]] = 1