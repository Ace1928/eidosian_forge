import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def all_idxs(self) -> Iterable[Tuple[int, ...]]:
    """Iterate over all possible indices of a multidimensional register."""
    yield from itertools.product(*[range(sh) for sh in self.shape])