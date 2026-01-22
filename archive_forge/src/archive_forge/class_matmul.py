import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
class matmul:

    def __init__(self, layout, block, mode, device, trans_a=False, trans_b=False, trans_c=False):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise NotImplementedError('Supported modes are: sdd, dsd, dds')
        self.block = block
        self.mode = mode
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.layout = layout
        self.spdims = layout.shape
        step = min(block, 32)
        if self.mode == 'sdd':
            self.c_lut, self.c_width = sdd_lut(layout, block, device)
            self.da_lut, self.da_width = dsd_lut(layout, block, step, True, device)
            self.db_lut, self.db_width = dsd_lut(layout, block, step, False, device)
        if self.mode == 'dsd':
            self.c_lut, self.c_width = dsd_lut(layout, block, step, not self.trans_a, device)
            self.da_lut, self.da_width = sdd_lut(layout, block, device)
            self.db_lut, self.db_width = dsd_lut(layout, block, step, self.trans_a, device)
        if self.mode == 'dds':
            self.c_lut, self.c_width = dsd_lut(layout, block, step, self.trans_b, device)
            self.da_lut, self.da_width = dsd_lut(layout, block, step, not self.trans_b, device)
            self.db_lut, self.db_width = sdd_lut(layout, block, device)

    def __call__(self, a, b, out=None):
        c = _matmul.apply(a, b, self.trans_a, self.trans_b, self.trans_c, self.mode, self.spdims, self.block, self.c_lut, self.c_width, self.da_lut, self.da_width, self.db_lut, self.db_width, out)
        return c