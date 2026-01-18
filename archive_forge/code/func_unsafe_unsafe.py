from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def unsafe_unsafe(self, a, b):
    """
        Set `a` can unsafe convert to `b` and `b` can unsafe convert to `a`
        """
    self._tg.unsafe(a, b)
    self._tg.unsafe(b, a)