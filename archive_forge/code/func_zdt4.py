import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def zdt4(individual):
    """ZDT4 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + 10(n-1) + \\sum_{i=2}^n \\left[ x_i^2 - 10\\cos(4\\pi x_i) \\right]`

    :math:`f_{\\text{ZDT4}1}(\\mathbf{x}) = x_1`

    :math:`f_{\\text{ZDT4}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[ 1 - \\sqrt{x_1/g(\\mathbf{x})} \\right]`

    """
    g = 1 + 10 * (len(individual) - 1) + sum((xi ** 2 - 10 * cos(4 * pi * xi) for xi in individual[1:]))
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return (f1, f2)