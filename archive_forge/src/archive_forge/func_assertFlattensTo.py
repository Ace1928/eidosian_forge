from __future__ import annotations
from typing import Type
from twisted.internet.defer import Deferred, succeed
from twisted.trial.unittest import SynchronousTestCase
from twisted.web import server
from twisted.web._flatten import flattenString
from twisted.web.error import FlattenerError
from twisted.web.http import Request
from twisted.web.resource import IResource
from twisted.web.template import Flattenable
from .requesthelper import DummyRequest
def assertFlattensTo(self, root: Flattenable, target: bytes) -> Deferred[bytes]:
    """
        Assert that a root element, when flattened, is equal to a string.
        """

    def check(result: bytes) -> bytes:
        self.assertEqual(result, target)
        return result
    d: Deferred[bytes] = flattenString(None, root)
    d.addCallback(check)
    return d