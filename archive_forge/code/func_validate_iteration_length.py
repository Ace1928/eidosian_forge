import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
@iteration_length.validator
def validate_iteration_length(self, attribute, value):
    if len(self.shape) != 0:
        raise ValueError(f'Selection register {self.name} should be flat. Found self.shape={self.shape!r}')
    if not 0 <= value <= 2 ** self.bitsize:
        raise ValueError(f'iteration length must be in range [0, 2^{self.bitsize}]')