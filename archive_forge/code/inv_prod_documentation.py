from cvxpy.atoms.elementwise.inv_pos import inv_pos
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.geo_mean import geo_mean
The reciprocal of a product of the entries of a vector ``x``.

    Parameters
    ----------
    x : Expression or numeric
        The expression whose reciprocal product is to be computed. Must have
        positive entries.

    Returns
    -------
    Expression
        .. math::
            \left(\prod_{i=1}^n x_i\right)^{-1},

        where :math:`n` is the length of :math:`x`.
    