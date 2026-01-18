import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_nonnegativity_transformation_1(self):
    self.model.a = Var()
    self.model.b = Var(within=NonNegativeIntegers)
    self.model.c = Var(within=Integers, bounds=(-2, 3))
    self.model.d = Var(within=Boolean)
    self.model.e = Var(domain=Boolean)
    instance = self.model.create_instance()
    xfrm = TransformationFactory('core.nonnegative_vars')
    transformed = xfrm.create_using(instance)
    for c in ('a', 'b', 'c', 'd', 'e'):
        var = transformed.__getattribute__(c)
        for ndx in var:
            self.assertTrue(self.nonnegativeBounds(var[ndx]))
    for ndx in transformed.a:
        self.assertIs(transformed.a[ndx].domain, NonNegativeReals)
    for ndx in transformed.b:
        self.assertIs(transformed.b[ndx].domain, NonNegativeIntegers)
    for ndx in transformed.c:
        self.assertIs(transformed.c[ndx].domain, NonNegativeIntegers)
    for ndx in transformed.d:
        self.assertIs(transformed.d[ndx].domain, Binary)
    for ndx in transformed.e:
        self.assertIs(transformed.e[ndx].domain, Binary)