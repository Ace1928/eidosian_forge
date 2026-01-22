import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
class TestTreeIgnores(TestCaseWithTransport):

    def assertPatternsEquals(self, patterns):
        with open('.bzrignore', 'rb') as f:
            contents = f.read().decode('utf-8').splitlines()
        self.assertEqual(sorted(patterns), sorted(contents))

    def test_new_file(self):
        tree = self.make_branch_and_tree('.')
        ignores.tree_ignores_add_patterns(tree, ['myentry'])
        self.assertTrue(tree.has_filename('.bzrignore'))
        self.assertPatternsEquals(['myentry'])

    def test_add_to_existing(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', b'myentry1\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2', 'foo'])
        self.assertPatternsEquals(['myentry1', 'myentry2', 'foo'])

    def test_adds_ending_newline(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', b'myentry1')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2'])
        self.assertPatternsEquals(['myentry1', 'myentry2'])
        with open('.bzrignore') as f:
            text = f.read()
        self.assertTrue(text.endswith(('\r\n', '\n', '\r')))

    def test_does_not_add_dupe(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', b'myentry\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry'])
        self.assertPatternsEquals(['myentry'])

    def test_non_ascii(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', 'myentryሴ\n'.encode())])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry噸'])
        self.assertPatternsEquals(['myentryሴ', 'myentry噸'])

    def test_crlf(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('.bzrignore', b'myentry1\r\n')])
        tree.add(['.bzrignore'])
        ignores.tree_ignores_add_patterns(tree, ['myentry2', 'foo'])
        with open('.bzrignore', 'rb') as f:
            self.assertEqual(f.read(), b'myentry1\r\nmyentry2\r\nfoo\r\n')
        self.assertPatternsEquals(['myentry1', 'myentry2', 'foo'])