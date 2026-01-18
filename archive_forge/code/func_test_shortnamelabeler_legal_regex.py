import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_shortnamelabeler_legal_regex(self):
    m = ConcreteModel()
    lbl = ShortNameLabeler(60, suffix='_', prefix='s_', legalRegex='^[a-zA-Z]')
    m.legal_var = Var()
    self.assertEqual(lbl(m.legal_var), 'legal_var')
    m._illegal_var = Var()
    self.assertEqual(lbl(m._illegal_var), 's__illegal_var_1_')