import enum
from .diff_with_sympy import differentiate as sympy_diff
from .diff_with_pyomo import reverse_sd, reverse_ad
Return derivative of expression.

    This function returns the derivative of expr with respect to one or
    more variables.  The type of the return value depends on the
    arguments wrt, wrt_list, and mode. See below for details.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
        The expression to differentiate
    wrt: pyomo.core.base.var._GeneralVarData
        If specified, this function will return the derivative with
        respect to wrt. wrt is normally a _GeneralVarData, but could
        also be a _ParamData. wrt and wrt_list cannot both be specified.
    wrt_list: list of pyomo.core.base.var._GeneralVarData
        If specified, this function will return the derivative with
        respect to each element in wrt_list.  A list will be returned
        where the values are the derivatives with respect to the
        corresponding entry in wrt_list.
    mode: pyomo.core.expr.calculus.derivatives.Modes
        Specifies the method to use for differentiation. Should be one
        of the members of the Modes enum:

            Modes.sympy:
                The pyomo expression will be converted to a sympy
                expression. Differentiation will then be done with
                sympy, and the result will be converted back to a pyomo
                expression.  The sympy mode only does symbolic
                differentiation. The sympy mode requires exactly one of
                wrt and wrt_list to be specified.
            Modes.reverse_symbolic:
                Symbolic differentiation will be performed directly with
                the pyomo expression in reverse mode. If neither wrt nor
                wrt_list are specified, then a ComponentMap is returned
                where there will be a key for each node in the
                expression tree, and the values will be the symbolic
                derivatives.
            Modes.reverse_numeric:
                Numeric differentiation will be performed directly with
                the pyomo expression in reverse mode. If neither wrt nor
                wrt_list are specified, then a ComponentMap is returned
                where there will be a key for each node in the
                expression tree, and the values will be the floating
                point values of the derivatives at the current values of
                the variables.

    Returns
    -------
    res: float, :py:class:`NumericExpression`, :py:class:`ComponentMap`, or list
        The value or expression of the derivative(s)

    