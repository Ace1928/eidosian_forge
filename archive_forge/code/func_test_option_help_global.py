import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_help_global(self):
    """Global options have help strings."""
    out, err = self.run_bzr('help status')
    self.assertContainsRe(out, '--show-ids.*Show internal object.')