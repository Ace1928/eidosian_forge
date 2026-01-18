from math import exp, sin, cos
def unwrapped_ball(data):
    """Unwrapped ball benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [-2, 8]^n`
       * - Function
         - :math:`f(\\mathbf{x}) = \\\\frac{10}{5 + \\sum_{i=1}^n (x_i - 3)^2}`
    """
    return 10.0 / (5.0 + sum(((d - 3) ** 2 for d in data)))