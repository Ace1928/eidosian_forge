import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestLookupAndFindPartialName(base.TestBase):
    scenarios = [('one-word', {'argv': ['o']}), ('two-words', {'argv': ['t', 'w']}), ('three-words', {'argv': ['t', 'w', 'c']})]

    def test(self):
        mgr = utils.TestCommandManager(utils.TEST_NAMESPACE)
        cmd, name, remaining = mgr.find_command(self.argv)
        self.assertTrue(cmd)
        self.assertEqual(' '.join(self.argv), name)
        self.assertFalse(remaining)