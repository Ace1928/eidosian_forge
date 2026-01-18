from __future__ import (absolute_import, division, print_function)
import pytest
@pytest.mark.skipif(pysym is None, reason='pysym missing')
def test_pysym_SymbolicSys_from_callback():
    _test(backend='pysym')