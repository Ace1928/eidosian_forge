import enum
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import Initializer
def _get_suffix_list(self, parent):
    if parent in self._suffixes_by_block:
        return self._suffixes_by_block[parent]
    suffixes = list(self._get_suffix_list(parent.parent_block()))
    self._suffixes_by_block[parent] = suffixes
    s = parent.component(self.name)
    if s is not None and s.ctype is Suffix and s.active:
        suffixes.append(s)
        self.all_suffixes.append(s)
    return suffixes