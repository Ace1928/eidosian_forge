import argparse
import functools
from cliff import command
from cliff.tests import base
def test_with_conflict_handler(self):
    cmd = TestCommand(None, None)
    cmd.conflict_handler = 'resolve'
    parser = cmd.get_parser('NAME')
    self.assertEqual(parser.conflict_handler, 'resolve')