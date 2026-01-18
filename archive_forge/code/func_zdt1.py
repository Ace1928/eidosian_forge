import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def zdt1(individual):
    """ZDT1 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`

    :math:`f_{\\text{ZDT1}1}(\\mathbf{x}) = x_1`

    :math:`f_{\\text{ZDT1}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\sqrt{\\frac{x_1}{g(\\mathbf{x})}}\\right]`
    """
    g = 1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1 / g))
    return (f1, f2)