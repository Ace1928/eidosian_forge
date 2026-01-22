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
def addfunc(name, f, _type, nargs, funcinfo, ae):
    if not isinstance(name, str):
        name = name.decode()
    self._known_functions[str(name)] = (f, _type, nargs, funcinfo, ae)