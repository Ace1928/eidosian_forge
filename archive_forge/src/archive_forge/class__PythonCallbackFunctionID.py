import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
class _PythonCallbackFunctionID(NumericConstant):
    """A specialized NumericConstant to preserve FunctionIDs through deepcopy.

    Function IDs are effectively pointers back to the
    PythonCallbackFunction.  As such, we need special handling to
    maintain / preserve the correct linkages through deepcopy (and
    model.clone()).

    """
    __slots__ = ()
    __autoslot_mappers__ = {'value': _python_callback_fid_mapper}

    def is_constant(self):
        return False