import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
def _run_cc(cmd, cwd, backend, log_stream=None):
    try:
        env = os.environ
        if _win32:
            extra_path = _get_extra_path_for_msvc()
            if extra_path is not None:
                path = extra_path + os.pathsep + os.environ.get('PATH', '')
                env = copy.deepcopy(env)
                env['PATH'] = path
        log = subprocess.check_output(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT, universal_newlines=True)
        if log_stream is not None:
            log_stream.write(log)
        return log
    except subprocess.CalledProcessError as e:
        msg = '`{0}` command returns non-zero exit status. \ncommand: {1}\nreturn-code: {2}\nstdout/stderr: \n{3}'.format(backend, e.cmd, e.returncode, e.output)
        if backend == 'nvcc':
            raise NVCCException(msg)
        elif backend == 'hipcc':
            raise HIPCCException(msg)
        else:
            raise RuntimeError(msg)
    except OSError as e:
        msg = 'Failed to run `{0}` command. Check PATH environment variable: ' + str(e)
        raise OSError(msg.format(backend))