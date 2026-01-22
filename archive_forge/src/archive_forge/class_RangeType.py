from .common import SimpleIterableType, SimpleIteratorType
from ..errors import TypingError
class RangeType(SimpleIterableType):

    def __init__(self, dtype):
        self.dtype = dtype
        name = 'range_state_%s' % (dtype,)
        super(SimpleIterableType, self).__init__(name)
        self._iterator_type = RangeIteratorType(self.dtype)

    def unify(self, typingctx, other):
        if isinstance(other, RangeType):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype is not None:
                return RangeType(dtype)