import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_suffix_rule(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    m.x = Var(m.I)
    m.y = Var(m.I)
    m.c = Constraint(m.I, rule=lambda m, i: m.x[i] >= i)
    m.d = Constraint(m.I, rule=lambda m, i: m.x[i] <= -i)
    _dict = {m.c[1]: 10, m.c[2]: 20, m.c[3]: 30, m.d: 100}
    m.suffix_dict = Suffix(initialize=_dict)
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_dict[m.c[1]], 10)
    self.assertEqual(m.suffix_dict[m.c[2]], 20)
    self.assertEqual(m.suffix_dict[m.c[3]], 30)
    self.assertEqual(m.suffix_dict[m.d[1]], 100)
    self.assertEqual(m.suffix_dict[m.d[2]], 100)
    self.assertEqual(m.suffix_dict[m.d[3]], 100)
    _dict[m.c[1]] = 1000
    m.suffix_dict.construct()
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_dict[m.c[1]], 10)
    m.suffix_cmap = Suffix(initialize=ComponentMap([(m.x[1], 10), (m.x[2], 20), (m.x[3], 30), (m.y, 100)]))
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_cmap[m.x[1]], 10)
    self.assertEqual(m.suffix_cmap[m.x[2]], 20)
    self.assertEqual(m.suffix_cmap[m.x[3]], 30)
    self.assertEqual(m.suffix_cmap[m.y[1]], 100)
    self.assertEqual(m.suffix_cmap[m.y[2]], 100)
    self.assertEqual(m.suffix_cmap[m.y[3]], 100)
    m.suffix_list = Suffix(initialize=[(m.x[1], 10), (m.x[2], 20), (m.x[3], 30), (m.y, 100)])
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_list[m.x[1]], 10)
    self.assertEqual(m.suffix_list[m.x[2]], 20)
    self.assertEqual(m.suffix_list[m.x[3]], 30)
    self.assertEqual(m.suffix_list[m.y[1]], 100)
    self.assertEqual(m.suffix_list[m.y[2]], 100)
    self.assertEqual(m.suffix_list[m.y[3]], 100)

    def gen_init():
        yield (m.x[1], 10)
        yield (m.x[2], 20)
        yield (m.x[3], 30)
        yield (m.y, 100)
    m.suffix_generator = Suffix(initialize=gen_init())
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_generator[m.x[1]], 10)
    self.assertEqual(m.suffix_generator[m.x[2]], 20)
    self.assertEqual(m.suffix_generator[m.x[3]], 30)
    self.assertEqual(m.suffix_generator[m.y[1]], 100)
    self.assertEqual(m.suffix_generator[m.y[2]], 100)
    self.assertEqual(m.suffix_generator[m.y[3]], 100)

    def genfcn_init(m, i):
        yield (m.x[1], 10)
        yield (m.x[2], 20)
        yield (m.x[3], 30)
        yield (m.y, 100)
    m.suffix_generator_fcn = Suffix(initialize=genfcn_init)
    self.assertEqual(len(m.suffix_dict), 6)
    self.assertEqual(m.suffix_generator_fcn[m.x[1]], 10)
    self.assertEqual(m.suffix_generator_fcn[m.x[2]], 20)
    self.assertEqual(m.suffix_generator_fcn[m.x[3]], 30)
    self.assertEqual(m.suffix_generator_fcn[m.y[1]], 100)
    self.assertEqual(m.suffix_generator_fcn[m.y[2]], 100)
    self.assertEqual(m.suffix_generator_fcn[m.y[3]], 100)