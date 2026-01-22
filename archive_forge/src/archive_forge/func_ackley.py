import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def ackley(individual):
    """Ackley test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-15, 30]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 20 - 20\\exp\\left(-0.2\\sqrt{\\frac{1}{N} \\
            \\sum_{i=1}^N x_i^2} \\right) + e - \\exp\\left(\\frac{1}{N}\\sum_{i=1}^N \\cos(2\\pi x_i) \\right)`

    .. plot:: code/benchmarks/ackley.py
       :width: 67 %
    """
    N = len(individual)
    return (20 - 20 * exp(-0.2 * sqrt(1.0 / N * sum((x ** 2 for x in individual)))) + e - exp(1.0 / N * sum((cos(2 * pi * x) for x in individual))),)