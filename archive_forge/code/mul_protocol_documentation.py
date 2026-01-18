from typing import Any
from cirq.protocols.resolve_parameters import is_parameterized
Returns lhs * rhs, or else a default if the operator is not implemented.

    This method is mostly used by __pow__ methods trying to return
    NotImplemented instead of causing a TypeError.

    Args:
        lhs: Left hand side of the multiplication.
        rhs: Right hand side of the multiplication.
        default: Default value to return if the multiplication is not defined.
            If not default is specified, a type error is raised when the
            multiplication fails.

    Returns:
        The product of the two inputs, or else the default value if the product
        is not defined, or else raises a TypeError if no default is defined.

    Raises:
        TypeError:
            lhs doesn't have __mul__ or it returned NotImplemented
            AND lhs doesn't have __rmul__ or it returned NotImplemented
            AND a default value isn't specified.
    