from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import pyomo.environ as pyo
from pyomo.repn.plugins.lp_writer import LPWriter
def test_warn_export_suffixes(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.obj = pyo.Objective(expr=m.x)
    m.con = pyo.Constraint(expr=m.x >= 2)
    m.b = pyo.Block()
    m.ignored = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.duals = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    m.b.duals = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    m.b.scaling = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    writer = LPWriter()
    with LoggingIntercept() as LOG:
        writer.write(m, StringIO())
    self.assertEqual(LOG.getvalue(), '')
    m.duals[m.con] = 5
    m.ignored[m.x] = 6
    m.b.scaling[m.x] = 7
    writer = LPWriter()
    with LoggingIntercept() as LOG:
        writer.write(m, StringIO())
    self.assertEqual(LOG.getvalue(), "EXPORT Suffix 'duals' found on 1 block:\n    duals\nLP writer cannot export suffixes to LP files.  Skipping.\nEXPORT Suffix 'scaling' found on 1 block:\n    b.scaling\nLP writer cannot export suffixes to LP files.  Skipping.\n")
    m.b.duals[m.x] = 7
    writer = LPWriter()
    with LoggingIntercept() as LOG:
        writer.write(m, StringIO())
    self.assertEqual(LOG.getvalue(), "EXPORT Suffix 'duals' found on 2 blocks:\n    duals\n    b.duals\nLP writer cannot export suffixes to LP files.  Skipping.\nEXPORT Suffix 'scaling' found on 1 block:\n    b.scaling\nLP writer cannot export suffixes to LP files.  Skipping.\n")