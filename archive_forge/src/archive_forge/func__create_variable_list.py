import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.matrix_constraint import matrix_constraint, _MatrixConstraintData
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression
from pyomo.core.kernel.block import block, block_list
def _create_variable_list(size, **kwds):
    assert size > 0
    vlist = variable_list()
    for i in range(size):
        vlist.append(variable(**kwds))
    return vlist