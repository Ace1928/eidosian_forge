import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestLookupAndFind(base.TestBase):
    scenarios = [('one-word', {'argv': ['one']}), ('two-words', {'argv': ['two', 'words']}), ('three-words', {'argv': ['three', 'word', 'command']})]

    def test(self):
        mgr = utils.TestCommandManager(utils.TEST_NAMESPACE)
        cmd, name, remaining = mgr.find_command(self.argv)
        self.assertTrue(cmd)
        self.assertEqual(' '.join(self.argv), name)
        self.assertFalse(remaining)