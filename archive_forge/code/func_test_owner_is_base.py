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
def test_owner_is_base(get_module):
    a = get_module.get_array_with_base()
    with pytest.warns(UserWarning, match='warn_on_free'):
        del a
        gc.collect()
        gc.collect()