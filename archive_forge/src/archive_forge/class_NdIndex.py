import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
@infer_global(pndindex)
@infer_global(np.ndindex)
class NdIndex(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1 and isinstance(args[0], types.BaseTuple):
            tup = args[0]
            if tup.count > 0 and (not isinstance(tup, types.UniTuple)):
                return
            shape = list(tup)
        else:
            shape = args
        if all((isinstance(x, types.Integer) for x in shape)):
            iterator_type = types.NumpyNdIndexType(len(shape))
            return signature(iterator_type, *args)