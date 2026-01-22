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
def _get_indexed_component_data_name(component, index):
    """Returns the fully-qualified component name for an unconstructed index.

    The ComponentData.name property assumes that the ComponentData has
    already been assigned to the owning Component.  This is a problem
    during the process of constructing a ComponentData instance, as we
    may need to throw an exception before the ComponentData is added to
    the owning component.  In those cases, we can use this function to
    generate the fully-qualified name without (permanently) adding the
    object to the Component.

    """
    if not component.is_indexed():
        return component.name
    elif index in component._data:
        ans = component._data[index].name
    else:
        for i in range(5):
            try:
                component._data[index] = component._ComponentDataClass(*(None,) * i, component=component)
                i = None
                break
            except:
                pass
        if i is not None:
            component._data[index] = component._ComponentDataClass(component=component)
        try:
            ans = component._data[index].name
        except:
            ans = component.name + '[{unknown index}]'
        finally:
            del component._data[index]
    return ans