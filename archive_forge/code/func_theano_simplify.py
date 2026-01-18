import logging
from sympy.external import import_module
from sympy.testing.pytest import raises, SKIP, warns_deprecated_sympy
import sympy as sy
from sympy.core.singleton import S
from sympy.abc import x, y, z, t
from sympy.printing.theanocode import (theano_code, dim_handling,
def theano_simplify(fgraph):
    """ Simplify a Theano Computation.

    Parameters
    ==========
    fgraph : theano.gof.FunctionGraph

    Returns
    =======
    theano.gof.FunctionGraph
    """
    mode = theano.compile.get_default_mode().excluding('fusion')
    fgraph = fgraph.clone()
    mode.optimizer.optimize(fgraph)
    return fgraph