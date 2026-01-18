import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_parse_args(self):
    """Option parser"""
    self.assertEqual(([], {'author': [], 'exclude': [], 'fixes': [], 'help': True, 'bugs': []}), parse_args(cmd_commit(), ['--help']))
    self.assertEqual(([], {'author': [], 'exclude': [], 'fixes': [], 'message': 'biter', 'bugs': []}), parse_args(cmd_commit(), ['--message=biter']))