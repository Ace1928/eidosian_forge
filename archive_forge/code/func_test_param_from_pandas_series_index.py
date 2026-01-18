import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
@unittest.skipUnless(pandas_available, 'pandas is not available')
def test_param_from_pandas_series_index(self):
    m = ConcreteModel()
    s = pd.Series([1, 3, 5], index=['T1', 'T2', 'T3'])
    m.I = Set(initialize=s.index)
    m.p1 = Param(m.I, initialize=s)
    self.assertEqual(m.p1.extract_values(), {'T1': 1, 'T2': 3, 'T3': 5})
    m.p2 = Param(s.index, initialize=s)
    self.assertEqual(m.p2.extract_values(), {'T1': 1, 'T2': 3, 'T3': 5})
    with self.assertRaisesRegex(KeyError, "Index 'T1' is not valid for indexed component 'p3'"):
        m.p3 = Param([0, 1, 2], initialize=s)
    m.J = Set(initialize=s)
    self.assertEqual(set(m.J), {1, 3, 5})