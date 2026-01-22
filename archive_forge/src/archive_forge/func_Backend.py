from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
def Backend(name=None, envvar='SYM_BACKEND', default='sympy'):
    """ Backend for the underlying symbolic manipulation packages

    Parameters
    ----------
    name: str (default: None)
        Name of package e.g. 'sympy'
    envvar: str (default: 'SYM_BACKEND')
        name of environment variable to read name from (when name is ``None``)
    default: str
        name to use when the environment variable described by ``envvar`` is
        unset or empty (default: 'sympy')

    Examples
    --------
    >>> be = Backend('sympy')  # or e.g. 'symengine'
    >>> x, y = map(be.Symbol, 'xy')
    >>> exprs = [x + y + 1, x*y**2]
    >>> lmb = be.Lambdify([x, y], exprs)
    >>> import numpy as np
    >>> lmb(np.arange(6.0).reshape((3, 2)))  # doctest: +NORMALIZE_WHITESPACE
    array([[   2.,    0.],
           [   6.,   18.],
           [  10.,  100.]])

    """
    if name is None:
        name = os.environ.get(envvar, '') or default
    if isinstance(name, _Base):
        return name
    else:
        return Backend.backends[name]()