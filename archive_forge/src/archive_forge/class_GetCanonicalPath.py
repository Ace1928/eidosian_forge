from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
class GetCanonicalPath(TestCaseWithTransport):

    def test_existing_case(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/b'])
        tree.add(['b'])
        self.assertEqual('b', get_canonical_path(tree, 'b', lambda x: x.lower()))
        self.assertEqual('b', get_canonical_path(tree, 'B', lambda x: x.lower()))

    def test_nonexistant_preserves_case(self):
        tree = self.make_branch_and_tree('tree')
        self.assertEqual('b', get_canonical_path(tree, 'b', lambda x: x.lower()))
        self.assertEqual('B', get_canonical_path(tree, 'B', lambda x: x.lower()))

    def test_in_directory_with_case(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a/', 'tree/a/b'])
        tree.add(['a', 'a/b'])
        self.assertEqual('a/b', get_canonical_path(tree, 'a/b', lambda x: x.lower()))
        self.assertEqual('a/b', get_canonical_path(tree, 'A/B', lambda x: x.lower()))
        self.assertEqual('a/b', get_canonical_path(tree, 'A/b', lambda x: x.lower()))
        self.assertEqual('a/C', get_canonical_path(tree, 'A/C', lambda x: x.lower()))

    def test_trailing_slash(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a/', 'tree/a/b'])
        tree.add(['a', 'a/b'])
        self.assertEqual('a', get_canonical_path(tree, 'a', lambda x: x))
        self.assertEqual('a', get_canonical_path(tree, 'a/', lambda x: x))