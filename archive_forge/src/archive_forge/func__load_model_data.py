import logging
import sys
from weakref import ref as weakref_ref
import gc
import math
from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.opt.results import Solution, SolverStatus, UndefinedData
from contextlib import nullcontext
from io import StringIO
def _load_model_data(self, modeldata, namespaces, **kwds):
    """
        Load declarations from a DataPortal object.
        """
    with PauseGC() as pgc:
        profile_memory = kwds.get('profile_memory', 0)
        if profile_memory >= 2 and pympler_available:
            mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
            print('')
            print('      Total memory = %d bytes prior to model construction' % mem_used)
            if profile_memory >= 3:
                gc.collect()
                mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
                print('      Total memory = %d bytes prior to model construction (after garbage collection)' % mem_used)
        for namespace in namespaces:
            if not namespace is None and (not namespace in modeldata._data):
                msg = "Cannot access undefined namespace: '%s'"
                raise IOError(msg % namespace)
        for component_name, component in self.component_map().items():
            if component.ctype is Model:
                continue
            self._initialize_component(modeldata, namespaces, component_name, profile_memory)
        if profile_memory >= 2 and pympler_available:
            print('')
            print('      Summary of objects following instance construction')
            post_construction_summary = pympler.summary.summarize(pympler.muppy.get_objects())
            pympler.summary.print_(post_construction_summary, limit=100)
            print('')