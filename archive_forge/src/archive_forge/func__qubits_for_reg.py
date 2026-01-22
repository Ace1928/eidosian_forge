import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _qubits_for_reg(reg: Register):
    if len(reg.shape) > 0:
        return _qubit_array(reg)
    return np.array([cirq.NamedQubit(f'{reg.name}')] if reg.total_bits() == 1 else cirq.NamedQubit.range(reg.total_bits(), prefix=reg.name), dtype=object)