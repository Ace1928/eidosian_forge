import logging
import math
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.base import _no_ctype, _convert_ctype
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.container_utils import define_simple_containers
@staticmethod
def _refresh_block_reserved_words():
    block._block_reserved_words = set(dir(block()))
    block._block_reserved_words.remove('active')