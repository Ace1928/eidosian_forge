from __future__ import annotations
from typing import TypeVar
from qiskit.quantum_info.operators.linear_op import LinearOp
Compute symmetric double commutator of a, b and c.

    See also Equation (13.6.18) in [1].

    If `commutator` is `True`, it returns

    .. math::

         [[A, B], C]/2 + [A, [B, C]]/2
         = (2ABC + 2CBA - BAC - CAB - ACB - BCA)/2.

    If `commutator` is `False`, it returns

    .. math::
         \lbrace[A, B], C\rbrace/2 + \lbrace A, [B, C]\rbrace/2
         = (2ABC - 2CBA - BAC + CAB - ACB + BCA)/2.

    Args:
        a: Operator a.
        b: Operator b.
        c: Operator c.
        commutator: If ``True`` compute the double commutator,
            if ``False`` the double anti-commutator.

    Returns:
        The double commutator

    References:

        [1]: R. McWeeny.
            Methods of Molecular Quantum Mechanics.
            2nd Edition, Academic Press, 1992.
            ISBN 0-12-486552-6.
    