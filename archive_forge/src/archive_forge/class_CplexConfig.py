from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.writers import LPWriter
import logging
import math
from pyomo.common.collections import ComponentMap
from typing import Optional, Sequence, NoReturn, List, Mapping, Dict
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
import sys
import time
from pyomo.common.log import LogStream
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager
class CplexConfig(MIPSolverConfig):

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super(CplexConfig, self).__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.declare('filename', ConfigValue(domain=str))
        self.declare('keepfiles', ConfigValue(domain=bool))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))
        self.filename = None
        self.keepfiles = False
        self.solver_output_logger = logger
        self.log_level = logging.INFO