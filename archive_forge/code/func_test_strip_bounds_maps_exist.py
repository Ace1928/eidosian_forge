import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_strip_bounds_maps_exist(self):
    """Tests if component maps for reversion already exist."""
    m = ConcreteModel()
    m.v0 = Var(bounds=(2, 4))
    m.v1 = Var(domain=NonNegativeReals)
    m.v2 = Var(domain=PositiveReals)
    m.v3 = Var(bounds=(-1, 1))
    m.v4 = Var(domain=Binary)
    m.v5 = Var(domain=Integers, bounds=(15, 16))
    xfrm = TransformationFactory('contrib.strip_var_bounds')
    xfrm.apply_to(m, reversible=True)
    with self.assertRaises(RuntimeError):
        xfrm.apply_to(m, reversible=True)