import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def total_bits(self) -> int:
    """The total number of bits in this register.

        This is the product of each of the dimensions in `shape`.
        """
    return self.bitsize * int(np.prod(self.shape))