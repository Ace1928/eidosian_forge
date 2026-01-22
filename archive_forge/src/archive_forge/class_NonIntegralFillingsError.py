from ..sage_helper import _within_sage, sage_method
from .. import snap
from . import exceptions
class NonIntegralFillingsError(RuntimeError):
    """
    Exception raised when Manifold has non-integral fillings, e.g.,
    for m004(1.1,1).
    """

    def __init__(self, manifold):
        self.manifold = manifold

    def __str__(self):
        return 'Manifold has non-integral Dehn-filings: %s' % self.manifold