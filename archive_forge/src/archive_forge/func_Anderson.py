from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
def Anderson(*args, **kwargs):
    """
    1d-solver generating pairs of approximative root and error.

    Uses Anderson-Bjoerk method to find a root of f in [a, b].
    Wrapper for illinois to use method='pegasus'.
    """
    kwargs['method'] = 'anderson'
    return Illinois(*args, **kwargs)