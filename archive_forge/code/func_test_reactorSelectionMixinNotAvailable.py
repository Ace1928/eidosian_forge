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
def test_reactorSelectionMixinNotAvailable(self):
    """
        Test that the usage mixin exits when trying to use a reactor not
        available (the reactor raises an error at installation), giving an
        error message.
        """

    class ReactorSelectionOptions(usage.Options, app.ReactorSelectionMixin):
        pass
    message = 'Missing foo bar'

    def install():
        raise ImportError(message)
    name = 'fakereactortest'
    package = __name__
    description = 'description'
    self.pluginResults = [FakeReactor(install, name, package, description)]
    options = ReactorSelectionOptions()
    options.messageOutput = StringIO()
    e = self.assertRaises(usage.UsageError, options.parseOptions, ['--reactor', 'fakereactortest', 'subcommand'])
    self.assertIn(message, e.args[0])
    self.assertIn('help-reactors', e.args[0])