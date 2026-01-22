from collections import defaultdict
from numba.core import config
class Rewrite(object):
    """Defines the abstract base class for Numba rewrites.
    """

    def __init__(self, state=None):
        """Constructor for the Rewrite class.
        """
        pass

    def match(self, func_ir, block, typemap, calltypes):
        """Overload this method to check an IR block for matching terms in the
        rewrite.
        """
        return False

    def apply(self):
        """Overload this method to return a rewritten IR basic block when a
        match has been found.
        """
        raise NotImplementedError('Abstract Rewrite.apply() called!')