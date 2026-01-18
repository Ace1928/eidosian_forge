import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
def test_suffix_warning(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var([1, 2, 3])
    m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
    m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
    m.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b = pyo.Block()
    m.b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    with LoggingIntercept() as LOG:
        repn = LinearStandardFormCompiler().write(m)
    self.assertEqual(LOG.getvalue(), '')
    m.dual[m.c] = 5
    with LoggingIntercept() as LOG:
        repn = LinearStandardFormCompiler().write(m)
    self.assertEqual(LOG.getvalue(), "EXPORT Suffix 'dual' found on 1 block:\n    dual\nStandard Form compiler ignores export suffixes.  Skipping.\n")
    m.b.dual[m.d] = 1
    with LoggingIntercept() as LOG:
        repn = LinearStandardFormCompiler().write(m)
    self.assertEqual(LOG.getvalue(), "EXPORT Suffix 'dual' found on 2 blocks:\n    dual\n    b.dual\nStandard Form compiler ignores export suffixes.  Skipping.\n")