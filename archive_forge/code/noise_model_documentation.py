from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
Noise model which replaces operations using a substitution function.

        Args:
            substitution_func: a function for replacing operations.
        