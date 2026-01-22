import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestFindInvalidCommand(base.TestBase):
    scenarios = [('no-such-command', {'argv': ['a', '-b']}), ('no-command-given', {'argv': ['-b']})]

    def test(self):
        mgr = utils.TestCommandManager(utils.TEST_NAMESPACE)
        try:
            mgr.find_command(self.argv)
        except ValueError as err:
            self.assertIn(self.argv[0], str(err))
            self.assertIn('-b', str(err))
        else:
            self.fail('expected a failure')