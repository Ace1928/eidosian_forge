import logging
import sys
from weakref import ref as weakref_ref
from pyomo.common.deprecation import deprecated, RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.misc import apply_indexed_rule
def _line_generator(k, v):
    for _k, _v in sorted(v.vars.items()):
        if _v is None:
            _val = '-'
        elif not hasattr(_v, 'is_indexed') or not _v.is_indexed():
            _val = str(value(_v))
        else:
            _val = '{%s}' % (', '.join(('%r: %r' % (x, value(_v[x])) for x in sorted(_v._data))),)
        yield (_k, _val)