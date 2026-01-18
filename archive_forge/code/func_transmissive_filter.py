from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def transmissive_filter(T):
    """An attenuator Jones matrix with transmittance ``T``.

    Parameters
    ==========

    T : numeric type or SymPy Symbol
        The transmittance of the attenuator.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import transmissive_filter
    >>> T = symbols("T", real=True)
    >>> NDF = transmissive_filter(T)
    >>> pprint(NDF, use_unicode=True)
    ⎡√T  0 ⎤
    ⎢      ⎥
    ⎣0   √T⎦

    """
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])