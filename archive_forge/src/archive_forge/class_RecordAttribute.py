import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer_getattr
class RecordAttribute(AttributeTemplate):
    key = types.Record

    def generic_resolve(self, record, attr):
        ret = record.typeof(attr)
        assert ret
        return ret