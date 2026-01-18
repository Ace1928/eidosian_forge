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
def test_substitute_casadi_intrinsic1(self):
    m = self.m
    m.y = Var()
    t = IndexTemplate(m.t)
    e = m.v[t]
    templatemap = {}
    e3 = substitute_pyomo2casadi(e, templatemap)
    self.assertIs(type(e3), casadi.SX)
    m.del_component('y')