import argparse
import functools
from cliff import command
from cliff.tests import base
def test_wrong_conflict_handler(self):
    cmd = TestCommand(None, None)
    cmd.conflict_handler = 'wrong'
    self.assertRaises(ValueError, cmd.get_parser, 'NAME')