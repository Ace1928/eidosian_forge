import abc
import logging
from scipy.sparse import coo_matrix
from pyomo.common.dependencies import numpy as np
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base import Var, Set, Constraint, value
from pyomo.core.base.block import _BlockData, Block, declare_custom_block
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.set import UnindexedComponent_set
from pyomo.core.base.reference import Reference
from ..sparse.block_matrix import BlockMatrix
class ScalarExternalGreyBoxBlock(ExternalGreyBoxBlockData, ExternalGreyBoxBlock):

    def __init__(self, *args, **kwds):
        ExternalGreyBoxBlockData.__init__(self, component=self)
        ExternalGreyBoxBlock.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index
    display = ExternalGreyBoxBlock.display