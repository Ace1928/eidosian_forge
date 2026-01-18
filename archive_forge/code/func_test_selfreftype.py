import re
import unittest
from wsme import exc
from wsme import types
def test_selfreftype(self):

    class SelfRefType(object):
        pass
    SelfRefType.parent = SelfRefType
    types.register_type(SelfRefType)