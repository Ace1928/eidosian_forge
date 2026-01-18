from numba.core.typeconv import castgraph, Conversion
from numba.core import types
def unsafe(self, a, b):
    """
        Set `a` can unsafe convert to `b`
        """
    self._tg.unsafe(a, b)