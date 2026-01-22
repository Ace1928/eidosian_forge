from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
def _generate_component_items(components):
    if type(components) not in {list, tuple}:
        components = (components,)
    for comp in components:
        if comp.is_indexed():
            for idx in sorted_robust(comp):
                yield (idx, comp[idx])
        else:
            yield (_NotAnIndex, comp)