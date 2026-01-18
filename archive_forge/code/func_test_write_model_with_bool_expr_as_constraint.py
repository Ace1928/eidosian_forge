import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
def test_write_model_with_bool_expr_as_constraint(self):
    m = ConcreteModel()
    m.i = IntervalVar([1, 2], optional=True)
    m.x = Var(within={1, 2})
    m.cons = LogicalConstraint(expr=m.i[m.x].is_present)
    cpx_mod, var_map = WriterFactory('docplex_model').write(m)
    variables = cpx_mod.get_all_variables()
    self.assertEqual(len(variables), 3)
    exprs = cpx_mod.get_all_expressions()
    self.assertEqual(len(exprs), 4)
    x = var_map[m.x]
    i1 = var_map[m.i[1]]
    i2 = var_map[m.i[2]]
    self.assertIs(variables[0], x)
    self.assertIs(variables[1], i2)
    self.assertIs(variables[2], i1)
    self.assertTrue(exprs[3][0].equals(cp.element([cp.presence_of(i1), cp.presence_of(i2)], 0 + 1 * (x - 1) // 1) == True))