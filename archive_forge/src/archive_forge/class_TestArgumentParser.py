import argparse
import functools
from cliff import command
from cliff.tests import base
class TestArgumentParser(base.TestBase):

    def test_option_name_collision(self):
        cmd = TestCommand(None, None)
        parser = cmd.get_parser('NAME')
        self.assertRaises(argparse.ArgumentError, parser.add_argument, '-z')

    def test_option_name_collision_with_alias(self):
        cmd = TestCommand(None, None)
        parser = cmd.get_parser('NAME')
        parser.add_argument('-z', '--zero')

    def test_resolve_option_with_name_collision(self):
        cmd = TestCommand(None, None)
        parser = cmd.get_parser('NAME')
        parser.add_argument('-z', '--zero', dest='zero', default='zero-default')
        args = parser.parse_args(['-z', 'foo', 'a', 'b'])
        self.assertEqual(args.zippy, 'foo')
        self.assertEqual(args.zero, 'zero-default')

    def test_with_conflict_handler(self):
        cmd = TestCommand(None, None)
        cmd.conflict_handler = 'resolve'
        parser = cmd.get_parser('NAME')
        self.assertEqual(parser.conflict_handler, 'resolve')

    def test_raise_conflict_argument_error(self):
        cmd = TestCommand(None, None)
        parser = cmd.get_parser('NAME')
        parser.add_argument('-f', '--foo', dest='foo', default='foo')
        self.assertRaises(argparse.ArgumentError, parser.add_argument, '-f')

    def test_resolve_conflict_argument(self):
        cmd = TestCommand(None, None)
        cmd.conflict_handler = 'resolve'
        parser = cmd.get_parser('NAME')
        parser.add_argument('-f', '--foo', dest='foo', default='foo')
        parser.add_argument('-f', '--foo', dest='foo', default='bar')
        args = parser.parse_args(['a', 'b'])
        self.assertEqual(args.foo, 'bar')

    def test_wrong_conflict_handler(self):
        cmd = TestCommand(None, None)
        cmd.conflict_handler = 'wrong'
        self.assertRaises(ValueError, cmd.get_parser, 'NAME')