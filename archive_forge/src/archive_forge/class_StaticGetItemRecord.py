import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer
class StaticGetItemRecord(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        record, idx = args
        if isinstance(record, types.Record) and isinstance(idx, str):
            if idx not in record.fields:
                raise KeyError(f"Field '{idx}' was not found in record with fields {tuple(record.fields.keys())}")
            ret = record.typeof(idx)
            assert ret
            return signature(ret, *args)