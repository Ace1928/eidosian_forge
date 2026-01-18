from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
Replaces all iterables in the OP_TREE with tuples.

    Args:
        root: The operation or tree of operations to freeze.

    Returns:
        An OP_TREE with the same operations and branching structure, but where
        all internal nodes are tuples instead of arbitrary iterables.
    