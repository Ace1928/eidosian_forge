from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
@infer_global(operator.sub)
@infer_global(operator.isub)
class DatetimeMinusTimedelta(AbstractTemplate):
    key = operator.sub

    def generic(self, args, kws):
        if len(args) == 1:
            return
        dt, td = args
        if isinstance(dt, types.NPDatetime) and isinstance(td, types.NPTimedelta):
            unit = npdatetime_helpers.combine_datetime_timedelta_units(dt.unit, td.unit)
            if unit is not None:
                return signature(types.NPDatetime(unit), dt, td)