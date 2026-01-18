from __future__ import (absolute_import, division, print_function)
import pytest
@pytest.mark.skipif(symengine is None, reason='symengine missing')
def test_symengine_SymbolicSys_from_callback():
    _test(backend='symengine')