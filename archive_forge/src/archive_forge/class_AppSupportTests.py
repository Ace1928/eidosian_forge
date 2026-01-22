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
class AppSupportTests(TestCase):

    def testPassphrase(self):
        self.assertIsNone(app.getPassphrase(0))

    def testLoadApplication(self):
        """
        Test loading an application file in different dump format.
        """
        a = service.Application('hello')
        baseconfig = {'file': None, 'source': None, 'python': None}
        for style in 'source pickle'.split():
            config = baseconfig.copy()
            config[{'pickle': 'file'}.get(style, style)] = 'helloapplication'
            sob.IPersistable(a).setStyle(style)
            sob.IPersistable(a).save(filename='helloapplication')
            a1 = app.getApplication(config, None)
            self.assertEqual(service.IService(a1).name, 'hello')
        config = baseconfig.copy()
        config['python'] = 'helloapplication'
        with open('helloapplication', 'w') as f:
            f.writelines(['from twisted.application import service\n', "application = service.Application('hello')\n"])
        a1 = app.getApplication(config, None)
        self.assertEqual(service.IService(a1).name, 'hello')

    def test_convertStyle(self):
        appl = service.Application('lala')
        for instyle in 'source pickle'.split():
            for outstyle in 'source pickle'.split():
                sob.IPersistable(appl).setStyle(instyle)
                sob.IPersistable(appl).save(filename='converttest')
                app.convertStyle('converttest', instyle, None, 'converttest.out', outstyle, 0)
                appl2 = service.loadApplication('converttest.out', outstyle)
                self.assertEqual(service.IService(appl2).name, 'lala')

    def test_startApplication(self):
        appl = service.Application('lala')
        app.startApplication(appl, 0)
        self.assertTrue(service.IService(appl).running)