import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_scan_subclass(self):

    class MyRoot(WSRoot):

        class SubClass(object):
            pass
    r = MyRoot()
    api = list(scan_api(r))
    assert len(api) == 0