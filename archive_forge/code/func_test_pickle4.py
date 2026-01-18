import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_pickle4(self):
    model = ConcreteModel()
    model.s = Set(initialize=[1, 2])
    model.x = Var(within=NonNegativeReals)
    model.x_indexed = Var(model.s, within=NonNegativeReals)
    model.obj = Objective(expr=model.x + model.x_indexed[1] + model.x_indexed[2])
    model.con = Constraint(expr=model.x >= 1)
    model.con2 = Constraint(expr=model.x_indexed[1] + model.x_indexed[2] >= 4)
    OUTPUT = open(join(currdir, 'test_pickle4_baseline.out'), 'w')
    model.pprint(ostream=OUTPUT)
    OUTPUT.close()
    _out, _txt = (join(currdir, 'test_pickle4_baseline.out'), join(currdir, 'test_pickle4_baseline.txt'))
    self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_out, _txt))
    str = pickle.dumps(model)
    OUTPUT = open(join(currdir, 'test_pickle4_after.out'), 'w')
    model.pprint(ostream=OUTPUT)
    OUTPUT.close()
    _out, _txt = (join(currdir, 'test_pickle4_after.out'), join(currdir, 'test_pickle4_baseline.txt'))
    self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_out, _txt))