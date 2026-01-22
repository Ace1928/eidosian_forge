from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
def _populate_generator_info(self):
    """
        Fill `index` for the Yield instruction and create YieldPoints.
        """
    dct = self.func_ir.generator_info.yield_points
    assert not dct, 'rerunning _populate_generator_info'
    for block in self.func_ir.blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                yieldinst = inst.value
                if isinstance(yieldinst, ir.Yield):
                    index = len(dct) + 1
                    yieldinst.index = index
                    yp = YieldPoint(block, yieldinst)
                    dct[yieldinst.index] = yp