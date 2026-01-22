import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def _test_disjuncts(self, blue_on=True):
    m = makeExpandedNetworkDisjunction()
    if blue_on:
        m.blue.indicator_var.fix(1)
        m.orange.indicator_var.fix(0)
    else:
        m.blue.indicator_var.fix(0)
        m.orange.indicator_var.fix(1)
    TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def initializer(blk):
        for _ in blk.component_data_objects(Constraint, active=True):
            SolverFactory('ipopt').solve(blk)
            break
    seq = SequentialDecomposition(select_tear_method='heuristic', default_guess=0.5)
    seq.run(m, initializer)
    if blue_on:
        self.assertAlmostEqual(value(m.dest.x), 0.84)
    else:
        self.assertAlmostEqual(value(m.dest.x), 0.42)