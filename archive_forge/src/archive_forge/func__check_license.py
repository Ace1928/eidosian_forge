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
def _check_license(self):
    if self._cplex_available:
        if not cmodel_available:
            Cplex._available = self.Availability.NeedsCompiledExtension
        else:
            try:
                m = self._cplex.Cplex()
                m.set_results_stream(None)
                m.variables.add(lb=[0] * 1001)
                m.solve()
                Cplex._available = self.Availability.FullLicense
            except self._cplex.exceptions.errors.CplexSolverError:
                try:
                    m = self._cplex.Cplex()
                    m.set_results_stream(None)
                    m.variables.add(lb=[0])
                    m.solve()
                    Cplex._available = self.Availability.LimitedLicense
                except:
                    Cplex._available = self.Availability.BadLicense
    else:
        Cplex._available = self.Availability.NotFound