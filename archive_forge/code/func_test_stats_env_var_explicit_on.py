import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
def test_stats_env_var_explicit_on(self):
    src = 'if 1:\n        from numba import njit\n        import numpy as np\n        from numba.core.runtime import rtsys, _nrt_python\n        from numba.core.registry import cpu_target\n\n        @njit\n        def foo():\n            return np.arange(10)[0]\n\n        # initialize the NRT before use\n        rtsys.initialize(cpu_target.target_context)\n        assert _nrt_python.memsys_stats_enabled()\n        orig_stats = rtsys.get_allocation_stats()\n        foo()\n        new_stats = rtsys.get_allocation_stats()\n        total_alloc = new_stats.alloc - orig_stats.alloc\n        total_free = new_stats.free - orig_stats.free\n        total_mi_alloc = new_stats.mi_alloc - orig_stats.mi_alloc\n        total_mi_free = new_stats.mi_free - orig_stats.mi_free\n\n        expected = 1\n        assert total_alloc == expected\n        assert total_free == expected\n        assert total_mi_alloc == expected\n        assert total_mi_free == expected\n        '
    env = os.environ.copy()
    env['NUMBA_NRT_STATS'] = '1'
    run_in_subprocess(src, env=env)