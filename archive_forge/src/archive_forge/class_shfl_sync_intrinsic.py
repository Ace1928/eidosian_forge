import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class shfl_sync_intrinsic(Stub):
    """
    shfl_sync_intrinsic(mask, mode, value, mode_offset, clamp)

    Nvvm intrinsic for shuffling data across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove
    """
    _description_ = '<shfl_sync()>'