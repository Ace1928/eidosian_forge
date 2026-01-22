from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@mod.validator
def _validate_mod(self, attribute, value):
    if not 1 <= value <= 2 ** self.bitsize:
        raise ValueError(f'mod: {value} must be between [1, {2 ** self.bitsize}].')