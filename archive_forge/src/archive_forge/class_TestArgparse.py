import unittest
from cliff import _argparse
class TestArgparse(unittest.TestCase):

    def test_argument_parser(self):
        _argparse.ArgumentParser(conflict_handler='ignore')

    def test_argument_parser_add_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        parser.add_argument_group()

    def test_argument_parser_add_mutually_exclusive_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        parser.add_mutually_exclusive_group()

    def test_argument_parser_add_nested_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        group = parser.add_argument_group()
        group.add_argument_group()

    def test_argument_parser_add_nested_mutually_exclusive_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        group = parser.add_argument_group()
        group.add_mutually_exclusive_group()

    def test_argument_parser_add_mx_nested_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        group = parser.add_mutually_exclusive_group()
        group.add_argument_group()

    def test_argument_parser_add_mx_nested_mutually_exclusive_group(self):
        parser = _argparse.ArgumentParser(conflict_handler='ignore')
        group = parser.add_mutually_exclusive_group()
        group.add_mutually_exclusive_group()