import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_custom_help(self):
    the_opt = option.Option.OPTIONS['help']
    orig_help = the_opt.help[:]
    my_opt = option.custom_help('help', 'suggest lottery numbers')
    self.assertEqual('suggest lottery numbers', my_opt.help)
    self.assertEqual(orig_help, the_opt.help)