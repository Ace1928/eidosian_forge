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
class ModelSolution(object):

    def __init__(self):
        self._metadata = {}
        self._metadata['status'] = None
        self._metadata['message'] = None
        self._metadata['gap'] = None
        self._entry = {}
        for name in ['objective', 'variable', 'constraint', 'problem']:
            self._entry[name] = {}

    def __getattr__(self, name):
        if name[0] == '_':
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
        return self.__dict__['_metadata'][name]

    def __setattr__(self, name, val):
        if name[0] == '_':
            self.__dict__[name] = val
            return
        self.__dict__['_metadata'][name] = val

    def __getstate__(self):
        state = {'_metadata': self._metadata, '_entry': {}}
        for name, data in self._entry.items():
            tmp = state['_entry'][name] = []
            for obj, entry in data.values():
                if obj is None or obj is None:
                    logger.warning("Solution component in '%s' no longer accessible: %s!" % (name, entry))
                else:
                    tmp.append((obj, entry))
        return state

    def __setstate__(self, state):
        self._metadata = state['_metadata']
        self._entry = {}
        for name, data in state['_entry'].items():
            tmp = self._entry[name] = {}
            for obj, entry in data:
                tmp[id(obj)] = (obj, entry)