import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
@property
def tunable_fields(self):
    out = ['XBLOCK', 'YBLOCK', 'ZBLOCK', 'RBLOCK', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'num_warps']
    if self.is_mm:
        out.append('num_stages')
    return out