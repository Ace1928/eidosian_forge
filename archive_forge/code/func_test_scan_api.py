import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_scan_api(self):

    class NS(object):

        @expose(int)
        @validate(int, int)
        def multiply(self, a, b):
            return a * b

    class MyRoot(WSRoot):
        ns = NS()
    r = MyRoot()
    api = list(scan_api(r))
    assert len(api) == 1
    path, fd, args = api[0]
    assert path == ['ns', 'multiply']
    assert fd._wsme_definition.name == 'multiply'
    assert args == []