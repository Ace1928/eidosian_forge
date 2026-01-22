import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
def _construct_from_rule_using_setitem(self):
    if self._rule is None:
        return
    index = None
    rule = self._rule
    block = self.parent_block()
    try:
        if rule.constant() and self.is_indexed():
            self._rule = rule = Initializer(rule(block, None), treat_sequences_as_mappings=False, arg_not_specified=NOTSET)
        if rule.contains_indices():
            for index in rule.indices():
                self[index] = rule(block, index)
        elif not self.index_set().isfinite():
            pass
        elif rule.constant():
            val = rule(block, None)
            for index in self.index_set():
                self._setitem_when_not_present(index, val)
        else:
            for index in self.index_set():
                self._setitem_when_not_present(index, rule(block, index))
    except:
        err = sys.exc_info()[1]
        logger.error("Rule failed for %s '%s' with index %s:\n%s: %s" % (self.ctype.__name__, self.name, str(index), type(err).__name__, err))
        raise