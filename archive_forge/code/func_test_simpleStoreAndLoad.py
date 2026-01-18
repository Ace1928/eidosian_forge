import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_simpleStoreAndLoad(self):
    a = service.Application('hello')
    p = sob.IPersistable(a)
    for style in 'source pickle'.split():
        p.setStyle(style)
        p.save()
        a1 = service.loadApplication('hello.ta' + style[0], style)
        self.assertEqual(service.IService(a1).name, 'hello')
    with open('hello.tac', 'w') as f:
        f.writelines(['from twisted.application import service\n', "application = service.Application('hello')\n"])
    a1 = service.loadApplication('hello.tac', 'python')
    self.assertEqual(service.IService(a1).name, 'hello')