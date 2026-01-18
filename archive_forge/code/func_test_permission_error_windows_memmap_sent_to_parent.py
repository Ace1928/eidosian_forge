import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr
@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_permission_error_windows_memmap_sent_to_parent(backend):
    cmd = "if 1:\n        import os\n        import time\n\n        import numpy as np\n\n        from joblib import Parallel, delayed\n        from testutils import return_slice_of_data\n\n        data = np.ones(int(2e6))\n\n        if __name__ == '__main__':\n            # warm-up call to launch the workers and start the resource_tracker\n            _ = Parallel(n_jobs=2, verbose=5, backend='{b}')(\n                delayed(id)(i) for i in range(20))\n\n            time.sleep(0.5)\n\n            slice_of_data = Parallel(n_jobs=2, verbose=5, backend='{b}')(\n                delayed(return_slice_of_data)(data, 0, 20) for _ in range(10))\n    ".format(b=backend)
    for _ in range(3):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(__file__)
        p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        p.wait()
        out, err = p.communicate()
        assert p.returncode == 0, err
        assert out == b''
        if sys.version_info[:3] not in [(3, 8, 0), (3, 8, 1)]:
            assert b'resource_tracker' not in err