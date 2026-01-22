from pyomo.common.autoslots import AutoSlots
from pyomo.core.base.block import _BlockData, IndexedBlock
from pyomo.core.base.global_set import UnindexedComponent_index, UnindexedComponent_set
class _TransformedDisjunct(IndexedBlock):
    _ComponentDataClass = _TransformedDisjunctData