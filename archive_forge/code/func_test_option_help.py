import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_help(self):
    """Options have help strings."""
    out, err = self.run_bzr('commit --help')
    self.assertContainsRe(out, '--file(.|\\n)*Take commit message from this file\\.')
    self.assertContainsRe(out, '-h.*--help')