import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols
def _with_row_col(self, row: int, col: int):
    return GridQubit(row, col)