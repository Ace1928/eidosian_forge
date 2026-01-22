from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(operator.delitem)
class DelItemSequence(AbstractTemplate):

    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            return signature(types.none, seq, idx)