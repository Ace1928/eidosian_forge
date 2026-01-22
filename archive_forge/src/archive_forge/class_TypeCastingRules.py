from numba.core.typeconv import castgraph, Conversion
from numba.core import types
class TypeCastingRules(object):
    """
    A helper for establishing type casting rules.
    """

    def __init__(self, tm):
        self._tm = tm
        self._tg = castgraph.TypeGraph(self._cb_update)

    def promote(self, a, b):
        """
        Set `a` can promote to `b`
        """
        self._tg.promote(a, b)

    def unsafe(self, a, b):
        """
        Set `a` can unsafe convert to `b`
        """
        self._tg.unsafe(a, b)

    def safe(self, a, b):
        """
        Set `a` can safe convert to `b`
        """
        self._tg.safe(a, b)

    def promote_unsafe(self, a, b):
        """
        Set `a` can promote to `b` and `b` can unsafe convert to `a`
        """
        self.promote(a, b)
        self.unsafe(b, a)

    def safe_unsafe(self, a, b):
        """
        Set `a` can safe convert to `b` and `b` can unsafe convert to `a`
        """
        self._tg.safe(a, b)
        self._tg.unsafe(b, a)

    def unsafe_unsafe(self, a, b):
        """
        Set `a` can unsafe convert to `b` and `b` can unsafe convert to `a`
        """
        self._tg.unsafe(a, b)
        self._tg.unsafe(b, a)

    def _cb_update(self, a, b, rel):
        """
        Callback for updating.
        """
        if rel == Conversion.promote:
            self._tm.set_promote(a, b)
        elif rel == Conversion.safe:
            self._tm.set_safe_convert(a, b)
        elif rel == Conversion.unsafe:
            self._tm.set_unsafe_convert(a, b)
        else:
            raise AssertionError(rel)