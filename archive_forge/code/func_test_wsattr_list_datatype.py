import re
import unittest
from wsme import exc
from wsme import types
def test_wsattr_list_datatype(self):
    import weakref
    a = types.wsattr(int)
    a.datatype = [weakref.ref(int)]
    assert isinstance(a.datatype, list)
    assert a.datatype[0] is int