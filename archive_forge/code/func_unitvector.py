from copy import copy
from ..libmp.backend import xrange
def unitvector(ctx, n, i):
    """
        Return the i-th n-dimensional unit vector.
        """
    assert 0 < i <= n, 'this unit vector does not exist'
    return [ctx.zero] * (i - 1) + [ctx.one] + [ctx.zero] * (n - i)