from pyomo.contrib.appsi.base import (
from pyomo.core.expr.numeric_expr import (
from pyomo.common.errors import PyomoException
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.numvalue import native_numeric_types
from typing import Dict, Optional, List
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.dependencies import attempt_import
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
import logging
import time
import sys
from pyomo.core.expr.visitor import ExpressionValueVisitor
class WntrConfig(SolverConfig):

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)