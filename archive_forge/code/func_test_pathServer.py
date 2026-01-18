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
@skipIf(not IReactorUNIX.providedBy(reactor), 'The reactor does not support UNIX domain sockets')
def test_pathServer(self) -> None:
    """
        The I{--path} option to L{makeService} causes it to return a service
        which will listen on the server address given by the I{--port} option.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    port = self.mktemp()
    options = Options()
    options.parseOptions(['--port', 'unix:' + port, '--path', path.path])
    service = makeService(options)
    service.startService()
    self.addCleanup(service.stopService)
    self.assertIsInstance(service.services[0].factory.resource, File)
    self.assertEqual(service.services[0].factory.resource.path, path.path)
    self.assertTrue(os.path.exists(port))
    self.assertTrue(stat.S_ISSOCK(os.stat(port).st_mode))