from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _add_dual_suffix(self, rHull):
    dual = rHull.component('dual')
    if dual is None:
        rHull.dual = Suffix(direction=Suffix.IMPORT)
    else:
        if dual.ctype is Suffix:
            return
        rHull.del_component(dual)
        rHull.dual = Suffix(direction=Suffix.IMPORT)
        rHull.add_component(unique_component_name(rHull, 'dual'), dual)