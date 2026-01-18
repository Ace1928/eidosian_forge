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
def test_rpyProcessor(self) -> None:
    """
        The I{--path} option creates a root resource which serves the
        C{resource} global defined by the Python source in any child with
        the C{".rpy"} extension.
        """
    path, root = self._pathOption()
    path.child('foo.rpy').setContent(b"from twisted.web.static import Data\nresource = Data('content', 'major/minor')\n")
    child = root.getChild('foo.rpy', None)
    self.assertIsInstance(child, Data)
    self.assertEqual(child.data, 'content')
    self.assertEqual(child.type, 'major/minor')