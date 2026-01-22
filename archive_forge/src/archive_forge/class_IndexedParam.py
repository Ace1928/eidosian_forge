import sys
import types
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression
class IndexedParam(Param):

    def __call__(self, exception=True):
        """Compute the value of the parameter"""
        if exception:
            raise TypeError('Cannot compute the value of an indexed Param (%s)' % (self.name,))

    def _create_objects_for_deepcopy(self, memo, component_list):
        if self.mutable:
            return super()._create_objects_for_deepcopy(memo, component_list)
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append(self)
        return _ans

    def __getitem__(self, args):
        try:
            return super().__getitem__(args)
        except:
            tmp = args if args.__class__ is tuple else (args,)
            if any((hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable() for arg in tmp)):
                return GetItemExpression((self,) + tmp)
            raise