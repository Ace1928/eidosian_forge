import abc
import enum
from typing import (
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData, Var
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from .utils.get_objective import get_objective
from .utils.collect_vars_and_named_exprs import collect_vars_and_named_exprs
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.common.errors import ApplicationError
from pyomo.opt.base import SolverFactory as LegacySolverFactory
from pyomo.common.factory import Factory
import os
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import (
from pyomo.opt.results.solver import (
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
import weakref
from .cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericConstant
class MIPSolverConfig(SolverConfig):
    """
    Attributes
    ----------
    mip_gap: float
        Solver will terminate if the mip gap is less than mip_gap
    relax_integrality: bool
        If True, all integer variables will be relaxed to continuous
        variables before solving
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super(MIPSolverConfig, self).__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.declare('mip_gap', ConfigValue(domain=NonNegativeFloat))
        self.declare('relax_integrality', ConfigValue(domain=bool))
        self.mip_gap: Optional[float] = None
        self.relax_integrality: bool = False