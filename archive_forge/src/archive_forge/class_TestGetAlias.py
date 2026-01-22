import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestGetAlias(tests.TestCase):

    def _get_config(self, config_text):
        my_config = config.GlobalConfig.from_string(config_text)
        return my_config

    def test_simple(self):
        my_config = self._get_config('[ALIASES]\ndiff=diff -r -2..-1\n')
        self.assertEqual(['diff', '-r', '-2..-1'], commands.get_alias('diff', config=my_config))

    def test_single_quotes(self):
        my_config = self._get_config("[ALIASES]\ndiff=diff -r -2..-1 --diff-options '--strip-trailing-cr -wp'\n")
        self.assertEqual(['diff', '-r', '-2..-1', '--diff-options', '--strip-trailing-cr -wp'], commands.get_alias('diff', config=my_config))

    def test_double_quotes(self):
        my_config = self._get_config('[ALIASES]\ndiff=diff -r -2..-1 --diff-options "--strip-trailing-cr -wp"\n')
        self.assertEqual(['diff', '-r', '-2..-1', '--diff-options', '--strip-trailing-cr -wp'], commands.get_alias('diff', config=my_config))

    def test_unicode(self):
        my_config = self._get_config('[ALIASES]\niam=whoami "Erik Bågfors <erik@bagfors.nu>"\n')
        self.assertEqual(['whoami', 'Erik Bågfors <erik@bagfors.nu>'], commands.get_alias('iam', config=my_config))