from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload
import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function
def _get_hash(val: Any, fail_quietly: bool=False):
    """Returns hash keys for caching a cirq.Operation and cirq.Gate.

    The hash of a cirq.Operation changes depending on its qubits, tags,
    classical controls, and other properties it has, none of these properties
    affect the TComplexity.
    For gates and gate backed operations we intend to compute the hash of the
    gate which is a property of the Gate.

    Args:
        val: object to compute its hash.

    Returns:
        - `val.gate` if `val` is a `cirq.Operation` which has an underlying `val.gate`.
        - `val` otherwise
    """
    if isinstance(val, cirq.Operation) and val.gate is not None:
        val = val.gate
    return val