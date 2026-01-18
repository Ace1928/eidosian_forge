from __future__ import (absolute_import, division, print_function)
import pytest
@pytest.mark.skipif(symcxx is None, reason='symcxx missing')
def test_symcxx_SymbolicSys_from_callback():
    _test(backend='symcxx')