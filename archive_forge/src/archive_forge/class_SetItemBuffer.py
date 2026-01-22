import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer_global(operator.setitem)
class SetItemBuffer(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args
        if not isinstance(ary, types.Buffer):
            return
        if not ary.mutable:
            msg = f'Cannot modify readonly array of type: {ary}'
            raise NumbaTypeError(msg)
        out = get_array_index_type(ary, idx)
        if out is None:
            return
        idx = out.index
        res = out.result
        if isinstance(res, types.Array):
            if isinstance(val, types.Array):
                if not self.context.can_convert(val.dtype, res.dtype):
                    return
                else:
                    res = val
            elif isinstance(val, types.Sequence):
                if res.ndim == 1 and self.context.can_convert(val.dtype, res.dtype):
                    res = val
                else:
                    return
            elif self.context.can_convert(val, res.dtype):
                res = res.dtype
            else:
                return
        elif not isinstance(val, types.Array):
            if not self.context.can_convert(val, res):
                if not res.is_precise():
                    newary = ary.copy(dtype=val)
                    return signature(types.none, newary, idx, res)
                else:
                    return
            res = val
        elif isinstance(val, types.Array) and val.ndim == 0 and self.context.can_convert(val.dtype, res):
            res = val
        else:
            return
        return signature(types.none, ary, idx, res)