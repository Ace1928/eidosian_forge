import argparse
import functools
from cliff import command
from cliff.tests import base
def test_option_name_collision(self):
    cmd = TestCommand(None, None)
    parser = cmd.get_parser('NAME')
    self.assertRaises(argparse.ArgumentError, parser.add_argument, '-z')