import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
def _get_magma_version():
    if 'Magma' not in torch.__config__.show():
        return (0, 0)
    position = torch.__config__.show().find('Magma ')
    version_str = torch.__config__.show()[position + len('Magma '):].split('\n')[0]
    return tuple((int(x) for x in version_str.split('.')))