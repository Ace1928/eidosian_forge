import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_set_short_name(self):
    o = option.Option('wiggle')
    o.set_short_name('w')
    self.assertEqual(o.short_name(), 'w')