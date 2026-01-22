import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _patch_lang(self, builder):
    lang = [value for _, value in self.fn.__globals__.items() if value in [tl, tl.core]]
    assert len(lang) == 1, "triton.language must be visible from within jit'd function"
    _patch_lang_tensor(getattr(lang[0], 'tensor'), builder)
    _patch_lang_core(lang[0], builder)