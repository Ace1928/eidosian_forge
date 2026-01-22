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
class HierarchicalModel(object):

    def __init__(self):
        m = self.model = ConcreteModel()
        m.a1_IDX = Set(initialize=[5, 4], ordered=True)
        m.a3_IDX = Set(initialize=[6, 7], ordered=True)
        m.c = Block()

        def x(b, i):
            pass

        def a(b, i):
            if i == 1:
                b.d = Block()
                b.c = Block(b.model().a1_IDX, rule=x)
            elif i == 3:
                b.e = Block()
                b.f = Block(b.model().a3_IDX, rule=x)
        m.a = Block([1, 2, 3], rule=a)
        m.b = Block()
        self.PrefixDFS = ['unknown', 'c', 'a[1]', 'a[1].d', 'a[1].c[5]', 'a[1].c[4]', 'a[2]', 'a[3]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'b']
        self.PrefixDFS_sortIdx = ['unknown', 'c', 'a[1]', 'a[1].d', 'a[1].c[4]', 'a[1].c[5]', 'a[2]', 'a[3]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'b']
        self.PrefixDFS_sortName = ['unknown', 'a[1]', 'a[1].c[5]', 'a[1].c[4]', 'a[1].d', 'a[2]', 'a[3]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'b', 'c']
        self.PrefixDFS_sort = ['unknown', 'a[1]', 'a[1].c[4]', 'a[1].c[5]', 'a[1].d', 'a[2]', 'a[3]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'b', 'c']
        self.PostfixDFS = ['c', 'a[1].d', 'a[1].c[5]', 'a[1].c[4]', 'a[1]', 'a[2]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'a[3]', 'b', 'unknown']
        self.PostfixDFS_sortIdx = ['c', 'a[1].d', 'a[1].c[4]', 'a[1].c[5]', 'a[1]', 'a[2]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'a[3]', 'b', 'unknown']
        self.PostfixDFS_sortName = ['a[1].c[5]', 'a[1].c[4]', 'a[1].d', 'a[1]', 'a[2]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'a[3]', 'b', 'c', 'unknown']
        self.PostfixDFS_sort = ['a[1].c[4]', 'a[1].c[5]', 'a[1].d', 'a[1]', 'a[2]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]', 'a[3]', 'b', 'c', 'unknown']
        self.BFS = ['unknown', 'c', 'a[1]', 'a[2]', 'a[3]', 'b', 'a[1].d', 'a[1].c[5]', 'a[1].c[4]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]']
        self.BFS_sortIdx = ['unknown', 'c', 'a[1]', 'a[2]', 'a[3]', 'b', 'a[1].d', 'a[1].c[4]', 'a[1].c[5]', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]']
        self.BFS_sortName = ['unknown', 'a[1]', 'a[2]', 'a[3]', 'b', 'c', 'a[1].c[5]', 'a[1].c[4]', 'a[1].d', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]']
        self.BFS_sort = ['unknown', 'a[1]', 'a[2]', 'a[3]', 'b', 'c', 'a[1].c[4]', 'a[1].c[5]', 'a[1].d', 'a[3].e', 'a[3].f[6]', 'a[3].f[7]']