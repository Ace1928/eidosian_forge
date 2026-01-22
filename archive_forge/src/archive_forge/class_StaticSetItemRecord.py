import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer
class StaticSetItemRecord(AbstractTemplate):
    key = 'static_setitem'

    def generic(self, args, kws):
        record, idx, value = args
        if isinstance(record, types.Record):
            if isinstance(idx, str):
                expectedty = record.typeof(idx)
                if self.context.can_convert(value, expectedty) is not None:
                    return signature(types.void, record, types.literal(idx), value)
            elif isinstance(idx, int):
                if idx >= len(record.fields):
                    msg = f'Requested index {idx} is out of range'
                    raise NumbaIndexError(msg)
                str_field = list(record.fields)[idx]
                expectedty = record.typeof(str_field)
                if self.context.can_convert(value, expectedty) is not None:
                    return signature(types.void, record, types.literal(idx), value)