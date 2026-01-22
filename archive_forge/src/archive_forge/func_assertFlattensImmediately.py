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
def assertFlattensImmediately(self, root: Flattenable, target: bytes) -> bytes:
    """
        Assert that a root element, when flattened, is equal to a string, and
        performs no asynchronus Deferred anything.

        This version is more convenient in tests which wish to make multiple
        assertions about flattening, since it can be called multiple times
        without having to add multiple callbacks.

        @return: the result of rendering L{root}, which should be equivalent to
            L{target}.
        @rtype: L{bytes}
        """
    return self.successResultOf(self.assertFlattensTo(root, target))