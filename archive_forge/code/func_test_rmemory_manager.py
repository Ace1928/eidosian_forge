import pytest
from rpy2 import rinterface
from rpy2.rinterface import memorymanagement
def test_rmemory_manager():
    with memorymanagement.rmemory() as rmemory:
        assert rmemory.count == 0
        foo = rmemory.protect(rinterface.conversion._str_to_charsxp('foo'))
        assert rmemory.count == 1
        del foo
    assert rmemory.count == 0