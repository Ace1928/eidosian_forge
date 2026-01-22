import os
import sys
from os.path import pardir, realpath
def _get_sysconfigdata_name():
    multiarch = getattr(sys.implementation, '_multiarch', '')
    return os.environ.get('_PYTHON_SYSCONFIGDATA_NAME', f'_sysconfigdata_{sys.abiflags}_{sys.platform}_{multiarch}')