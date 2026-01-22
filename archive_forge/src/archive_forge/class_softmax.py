import torch
from ... import jit
from ... import language as tl
from ... import next_power_of_2
class softmax:

    def __init__(self, layout, block, device, is_dense=False):
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.lut, self.maxlut = _softmax.make_lut(self.layout, self.block, device)
        self.is_dense = is_dense

    def __call__(self, a, *, scale=1.0, rel_logits=None, is_causal=False):
        if rel_logits is not None and rel_logits.dtype != a.dtype:
            raise ValueError(f'relative position embedding must be {a.dtype}')
        a = _softmax.apply(a, scale, rel_logits, is_causal, self.spdims, self.block, self.lut, self.maxlut, self.is_dense)
        return a