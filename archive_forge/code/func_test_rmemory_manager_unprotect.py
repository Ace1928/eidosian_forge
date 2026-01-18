import pytest
from rpy2 import rinterface
from rpy2.rinterface import memorymanagement
def test_rmemory_manager_unprotect():
    with memorymanagement.rmemory() as rmemory:
        assert rmemory.count == 0
        foo = rmemory.protect(rinterface.conversion._str_to_charsxp('foo'))
        with pytest.raises(ValueError):
            rmemory.unprotect(2)
        rmemory.unprotect(1)
        assert rmemory.count == 0
        del foo
    assert rmemory.count == 0