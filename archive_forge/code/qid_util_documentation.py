from typing import overload, TYPE_CHECKING, Union
Constructs a qubit id of the appropriate type based on args.

    This is shorthand for constructing qubit ids of common types:
    >>> cirq.q(1) == cirq.LineQubit(1)
    True
    >>> cirq.q(1, 2) == cirq.GridQubit(1, 2)
    True
    >>> cirq.q("foo") == cirq.NamedQubit("foo")
    True

    Note that arguments should be treated as positional only, even
    though this is only enforceable in python 3.8 or later.

    Args:
        *args: One or two ints, or a single str, as described above.

    Returns:
        cirq.LineQubit if called with one integer arg.
        cirq.GridQubit if called with two integer args.
        cirq.NamedQubit if called with one string arg.

    Raises:
        ValueError: if called with invalid arguments.
    