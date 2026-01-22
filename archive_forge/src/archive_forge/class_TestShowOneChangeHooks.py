from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
class TestShowOneChangeHooks(base.TestBase):

    def setUp(self):
        super(TestShowOneChangeHooks, self).setUp()
        self.app = make_app()
        self.cmd = TestShowCommand(self.app, None, cmd_name='test')
        self.hook = TestDisplayChangeHook(self.cmd)
        self.mgr = extension.ExtensionManager.make_test_instance([extension.Extension('parser-hook', None, None, self.hook)])
        self.cmd._hooks = self.mgr

    def test_get_parser(self):
        parser = self.cmd.get_parser('test')
        results = parser.parse_args(['--added-by-hook', 'value'])
        self.assertEqual(results.added_by_hook, 'value')

    def test_get_epilog(self):
        results = self.cmd.get_epilog()
        self.assertIn('hook epilog', results)

    def test_before(self):
        self.assertFalse(self.hook._before_called)
        parser = self.cmd.get_parser('test')
        results = parser.parse_args(['--added-by-hook', 'value'])
        self.cmd.run(results)
        self.assertTrue(self.hook._before_called)
        self.assertEqual(results.added_by_hook, 'othervalue')
        self.assertTrue(results.added_by_before)

    def test_after(self):
        self.assertFalse(self.hook._after_called)
        parser = self.cmd.get_parser('test')
        results = parser.parse_args(['--added-by-hook', 'value'])
        result = self.cmd.run(results)
        self.assertTrue(self.hook._after_called)
        self.assertEqual(result, 0)