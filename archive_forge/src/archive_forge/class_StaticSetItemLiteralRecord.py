import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer_global(operator.setitem)
class StaticSetItemLiteralRecord(AbstractTemplate):

    def generic(self, args, kws):
        target, idx, value = args
        if isinstance(target, types.Record) and isinstance(idx, types.StringLiteral):
            if idx.literal_value not in target.fields:
                msg = f"Field '{idx.literal_value}' was not found in record with fields {tuple(target.fields.keys())}"
                raise NumbaKeyError(msg)
            expectedty = target.typeof(idx.literal_value)
            if self.context.can_convert(value, expectedty) is not None:
                return signature(types.void, target, idx, value)