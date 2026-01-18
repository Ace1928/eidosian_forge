import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_arg_help(self):
    """Help message shows option arguments."""
    out, err = self.run_bzr('help commit')
    self.assertEqual(err, '')
    self.assertContainsRe(out, '--file[ =]MSGFILE')