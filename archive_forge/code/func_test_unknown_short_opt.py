import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_unknown_short_opt(self):
    out, err = self.run_bzr('help -r', retcode=3)
    self.assertContainsRe(err, 'no such option')