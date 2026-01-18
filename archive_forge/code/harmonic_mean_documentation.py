from cvxpy.atoms.pnorm import pnorm
from cvxpy.expressions.expression import Expression
The harmonic mean of ``x``.

    Parameters
    ----------
    x : Expression or numeric
        The expression whose harmonic mean is to be computed. Must have
        positive entries.

    Returns
    -------
    Expression
        .. math::
            \frac{n}{\left(\sum_{i=1}^{n} x_i^{-1} \right)},

        where :math:`n` is the length of :math:`x`.
    