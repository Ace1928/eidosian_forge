import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_single_quotes(self):
    my_config = self._get_config("[ALIASES]\ndiff=diff -r -2..-1 --diff-options '--strip-trailing-cr -wp'\n")
    self.assertEqual(['diff', '-r', '-2..-1', '--diff-options', '--strip-trailing-cr -wp'], commands.get_alias('diff', config=my_config))