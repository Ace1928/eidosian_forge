from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
class TestCommandLoadHooks(base.TestBase):

    def test_no_app_or_name(self):
        cmd = TestCommand(None, None)
        self.assertEqual([], cmd._hooks)

    @mock.patch('stevedore.extension.ExtensionManager')
    def test_app_and_name(self, em):
        app = make_app()
        TestCommand(app, None, cmd_name='test')
        print(em.mock_calls[0])
        name, args, kwargs = em.mock_calls[0]
        print(kwargs)
        self.assertEqual('cliff.tests.test', kwargs['namespace'])