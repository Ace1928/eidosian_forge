import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
class ModelSOS(object):

    class AmplSuffix(object):

        def __init__(self, name):
            self.name = name
            self.ids = []
            self.vals = []

        def add(self, idx, val):
            if idx in self.ids:
                raise RuntimeError('The NL file format does not support multiple nonzero values for a single component and suffix. \nSuffix Name:  %s\nComponent ID: %s\n' % (self.name, idx))
            else:
                self.ids.append(idx)
                self.vals.append(val)

        def genfilelines(self):
            base_line = '{0} {1}\n'
            return [base_line.format(idx, val) for idx, val in zip(self.ids, self.vals) if val != 0]

        def is_empty(self):
            return not bool(len(self.ids))

    def __init__(self, ampl_var_id, varID_map):
        self.ampl_var_id = ampl_var_id
        self.sosno = self.AmplSuffix('sosno')
        self.ref = self.AmplSuffix('ref')
        self.block_cntr = 0
        self.varID_map = varID_map

    def count_constraint(self, soscondata):
        ampl_var_id = self.ampl_var_id
        varID_map = self.varID_map
        if hasattr(soscondata, 'get_items'):
            sos_items = list(soscondata.get_items())
        else:
            sos_items = list(soscondata.items())
        if len(sos_items) == 0:
            return
        level = soscondata.level
        self.block_cntr += 1
        sign_tag = None
        if level == 1:
            sign_tag = 1
        elif level == 2:
            sign_tag = -1
        else:
            raise ValueError("SOSConstraint '%s' has sos type='%s', which is not supported by the NL file interface" % (soscondata.name, level))
        for vardata, weight in sos_items:
            weight = _get_bound(weight)
            if weight < 0:
                raise ValueError('Cannot use negative weight %f for variable %s is special ordered set %s ' % (weight, vardata.name, soscondata.name))
            if vardata.fixed:
                raise ValueError("SOSConstraint '%s' includes a fixed Variable '%s'. This is currently not supported. Deactivate this constraint in order to proceed" % (soscondata.name, vardata.name))
            ID = ampl_var_id[varID_map[id(vardata)]]
            self.sosno.add(ID, self.block_cntr * sign_tag)
            self.ref.add(ID, weight)