from typing import Any
Assert that the given object has a `_repr_pretty_` output that contains the given text.

    Args:
            val: The object to test.
            substr: The string that `_repr_pretty_` is expected to contain.
            cycle: The value of `cycle` passed to `_repr_pretty_`.  `cycle` represents whether
                the call is made with a potential cycle. Typically one should handle the
                `cycle` equals `True` case by returning text that does not recursively call
                the `_repr_pretty_` to break this cycle.

    Raises:
        AssertionError: If `_repr_pretty_` does not pretty print the given text.
    