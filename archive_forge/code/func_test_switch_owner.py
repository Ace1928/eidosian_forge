import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
@pytest.mark.xfail(sys.implementation.name == 'pypy', reason='bad interaction between getenv and os.environ inside pytest')
@pytest.mark.parametrize('policy', ['0', '1', None])
def test_switch_owner(get_module, policy):
    a = get_module.get_array()
    assert np.core.multiarray.get_handler_name(a) is None
    get_module.set_own(a)
    if policy is None:
        policy = os.getenv('NUMPY_WARN_IF_NO_MEM_POLICY', '0') == '1'
        oldval = None
    else:
        policy = policy == '1'
        oldval = np.core._multiarray_umath._set_numpy_warn_if_no_mem_policy(policy)
    try:
        if policy:
            with assert_warns(RuntimeWarning) as w:
                del a
                gc.collect()
        else:
            del a
            gc.collect()
    finally:
        if oldval is not None:
            np.core._multiarray_umath._set_numpy_warn_if_no_mem_policy(oldval)