import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
def test_substitute_casadi_sym(self):
    m = self.m
    m.y = Var()
    t = IndexTemplate(m.t)
    e = m.dv[t] + m.v[t] + m.y + t
    templatemap = {}
    e2 = substitute_pyomo2casadi(e, templatemap)
    self.assertEqual(len(templatemap), 2)
    self.assertIs(type(e2.arg(0)), casadi.SX)
    self.assertIs(type(e2.arg(1)), casadi.SX)
    self.assertIsNot(type(e2.arg(2)), casadi.SX)
    self.assertIs(type(e2.arg(3)), IndexTemplate)
    m.del_component('y')