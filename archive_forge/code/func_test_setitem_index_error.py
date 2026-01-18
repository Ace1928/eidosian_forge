import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_setitem_index_error(self):
    try:
        self.instance.A[2] = 4.3
        if not self.instance.A.mutable:
            self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
        self.fail('Expected KeyError because 2 is not a valid key')
    except KeyError:
        pass
    except TypeError:
        if self.instance.A.mutable:
            raise