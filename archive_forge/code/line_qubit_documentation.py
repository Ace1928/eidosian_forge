import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
Returns a range of line qubits.

        Args:
            *range_args: Same arguments as python's built-in range method.

        Returns:
            A list of line qubits.
        