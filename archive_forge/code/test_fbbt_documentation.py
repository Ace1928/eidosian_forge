import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.fileutils import find_library
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (
import math
import platform
from io import StringIO

    These tests are set up weird, but it is for a good reason.
    The FBBT code is duplicated in pyomo.contrib.appsi for
    improved performance. We want to keep this version because
    it does not require building an extension. However, when we
    fix a bug in one module, we want to ensure we fix that bug
    in the other module. Therefore, we use this base class
    for testing both modules. The only difference in the
    derived classes is self.tightener attribute.
    