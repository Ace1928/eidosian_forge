from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_pseudomap_getitem_exceptionString(self):

    def tester(pm, _str):
        try:
            pm['a']
            self.fail('Expected PseudoMap to raise a KeyError')
        except KeyError:
            err = sys.exc_info()[1].args[0]
            self.assertEqual(_str, err)
    m = Block(name='foo')
    tester(m.component_map(), "component 'a' not found in block foo")
    tester(m.component_map(active=True), "active component 'a' not found in block foo")
    tester(m.component_map(active=False), "inactive component 'a' not found in block foo")
    tester(m.component_map(Var), "Var component 'a' not found in block foo")
    tester(m.component_map(Var, active=True), "active Var component 'a' not found in block foo")
    tester(m.component_map(Var, active=False), "inactive Var component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var)), "SubclassOf(Var) component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var), active=True), "active SubclassOf(Var) component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var), active=False), "inactive SubclassOf(Var) component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var, Block)), "SubclassOf(Var,Block) component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var, Block), active=True), "active SubclassOf(Var,Block) component 'a' not found in block foo")
    tester(m.component_map(SubclassOf(Var, Block), active=False), "inactive SubclassOf(Var,Block) component 'a' not found in block foo")
    tester(m.component_map([Var, Param]), "Param or Var component 'a' not found in block foo")
    tester(m.component_map(set([Var, Param]), active=True), "active Param or Var component 'a' not found in block foo")
    tester(m.component_map(set([Var, Param]), active=False), "inactive Param or Var component 'a' not found in block foo")
    tester(m.component_map(set([Set, Var, Param])), "Param, Set or Var component 'a' not found in block foo")
    tester(m.component_map(set([Set, Var, Param]), active=True), "active Param, Set or Var component 'a' not found in block foo")
    tester(m.component_map(set([Set, Var, Param]), active=False), "inactive Param, Set or Var component 'a' not found in block foo")