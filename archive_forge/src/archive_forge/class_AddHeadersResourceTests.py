from __future__ import annotations
import os
import stat
from typing import cast
from unittest import skipIf
from twisted.internet import endpoints, reactor
from twisted.internet.interfaces import IReactorCore, IReactorUNIX
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.threadpool import ThreadPool
from twisted.python.usage import UsageError
from twisted.spread.pb import PBServerFactory
from twisted.trial.unittest import TestCase
from twisted.web import demo
from twisted.web.distrib import ResourcePublisher, UserDirectory
from twisted.web.script import PythonScript
from twisted.web.server import Site
from twisted.web.static import Data, File
from twisted.web.tap import (
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.twcgi import CGIScript
from twisted.web.wsgi import WSGIResource
class AddHeadersResourceTests(TestCase):

    def test_getChildWithDefault(self) -> None:
        """
        When getChildWithDefault is invoked, it adds the headers to the
        response.
        """
        resource = _AddHeadersResource(demo.Test(), [('K1', 'V1'), ('K2', 'V2'), ('K1', 'V3')])
        request = DummyRequest([])
        resource.getChildWithDefault('', request)
        self.assertEqual(request.responseHeaders.getRawHeaders('K1'), ['V1', 'V3'])
        self.assertEqual(request.responseHeaders.getRawHeaders('K2'), ['V2'])