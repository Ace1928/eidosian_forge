import argparse
import functools
from cliff import command
from cliff.tests import base
def test_raise_conflict_argument_error(self):
    cmd = TestCommand(None, None)
    parser = cmd.get_parser('NAME')
    parser.add_argument('-f', '--foo', dest='foo', default='foo')
    self.assertRaises(argparse.ArgumentError, parser.add_argument, '-f')