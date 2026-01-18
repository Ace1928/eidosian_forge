import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
@m.Constraint([1, 2, 3])
def x_bounds(m, i):
    return m.x[i] >= 3 * (i - 1)