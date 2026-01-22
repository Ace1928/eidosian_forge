from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(operator.getitem)
class GetItemSequence(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.Sequence):
            idx = normalize_1d_index(idx)
            if isinstance(idx, types.SliceType):
                if not isinstance(seq, types.BaseTuple):
                    return signature(seq, seq, idx)
            elif isinstance(idx, types.Integer):
                return signature(seq.dtype, seq, idx)