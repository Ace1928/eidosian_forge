from __future__ import annotations
import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)
def _get_num_warps_from_ir_str(src: str):
    num_warps_matches = re.findall(ttgir_num_warps_pattern, src)
    assert len(num_warps_matches) == 1, 'Expected exactly one match for num_warps'
    num_warps = int(num_warps_matches[0])
    num_warp_groups_matches = re.findall('"triton_gpu.num-warp-groups-per-cta"\\s?=\\s?(\\d+)\\s?:', src)
    assert len(num_warp_groups_matches) == 0 or len(num_warp_groups_matches) == 1, 'Expected triton_gpu.num-warp-groups-per-cta attribute to appear 0 or 1 times'
    if num_warp_groups_matches:
        num_warps *= int(num_warp_groups_matches[0])
    return num_warps