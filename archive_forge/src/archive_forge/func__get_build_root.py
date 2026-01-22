import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any, List, Optional
import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension
def _get_build_root() -> str:
    global _BUILD_ROOT
    if _BUILD_ROOT is None:
        _BUILD_ROOT = _make_temp_dir(prefix='benchmark_utils_jit_build')
        atexit.register(shutil.rmtree, _BUILD_ROOT)
    return _BUILD_ROOT