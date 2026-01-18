import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from cirq import protocols
from cirq.ops import raw_types
Returns a range of `cirq.NamedQubit`s.

        The range returned starts with the prefix, and followed by a qubit for
        each number in the range, e.g.:

            >>> cirq.NamedQubit.range(3, prefix='a')
            ... # doctest: +NORMALIZE_WHITESPACE
            [cirq.NamedQubit('a0'), cirq.NamedQubit('a1'),
                cirq.NamedQubit('a2')]
            >>> cirq.NamedQubit.range(2, 4, prefix='a')
            [cirq.NamedQubit('a2'), cirq.NamedQubit('a3')]

        Args:
            *args: Args to be passed to Python's standard range function.
            prefix: A prefix for constructed NamedQubits.

        Returns:
            A list of ``NamedQubit``\\s.
        