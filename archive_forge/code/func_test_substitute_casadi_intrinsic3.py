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
def test_substitute_casadi_intrinsic3(self):
    m = self.m
    m.y = Var()
    t = IndexTemplate(m.t)
    e = sin(m.dv[t] + m.v[t]) + log(m.v[t] * m.y + m.dv[t] ** 2)
    templatemap = {}
    e3 = substitute_pyomo2casadi(e, templatemap)
    self.assertIs(e3.arg(0)._fcn, casadi.sin)
    self.assertIs(e3.arg(1)._fcn, casadi.log)
    m.del_component('y')