import heapq
import torch
from .. import cdiv
from .._C.libtriton.triton import runtime
from ..runtime import driver
from ..testing import (get_dram_gbps, get_max_simd_tflops, get_max_tensorcore_tflops, nvsmi)
 return estimated running time in ms
          = max(compute, loading) + store 