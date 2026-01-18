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
def test_wsgi(self) -> None:
    """
        The I{--wsgi} option takes the fully-qualifed Python name of a WSGI
        application object and creates a L{WSGIResource} at the root which
        serves that application.
        """
    options = Options()
    options.parseOptions(['--wsgi', __name__ + '.application'])
    root = options['root']
    self.assertTrue(root, WSGIResource)
    self.assertIdentical(root._reactor, reactor)
    self.assertTrue(isinstance(root._threadpool, ThreadPool))
    self.assertIdentical(root._application, application)
    self.assertFalse(root._threadpool.started)
    cast(IReactorCore, reactor).fireSystemEvent('startup')
    self.assertTrue(root._threadpool.started)
    self.assertFalse(root._threadpool.joined)
    cast(IReactorCore, reactor).fireSystemEvent('shutdown')
    self.assertTrue(root._threadpool.joined)