import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class TestBisect(TestCaseWithDirState):
    """Test the ability to bisect into the disk format."""

    def assertBisect(self, expected_map, map_keys, state, paths):
        """Assert that bisecting for paths returns the right result.

        :param expected_map: A map from key => entry value
        :param map_keys: The keys to expect for each path
        :param state: The DirState object.
        :param paths: A list of paths, these will automatically be split into
                      (dir, name) tuples, and sorted according to how _bisect
                      requires.
        """
        result = state._bisect(paths)
        self.assertEqual(len(map_keys), len(paths))
        expected = {}
        for path, keys in zip(paths, map_keys):
            if keys is None:
                continue
            expected[path] = sorted((expected_map[k] for k in keys))
        for path in result:
            result[path].sort()
        self.assertEqual(expected, result)

    def assertBisectDirBlocks(self, expected_map, map_keys, state, paths):
        """Assert that bisecting for dirbblocks returns the right result.

        :param expected_map: A map from key => expected values
        :param map_keys: A nested list of paths we expect to be returned.
            Something like [['a', 'b', 'f'], ['b/c', 'b/d']]
        :param state: The DirState object.
        :param paths: A list of directories
        """
        result = state._bisect_dirblocks(paths)
        self.assertEqual(len(map_keys), len(paths))
        expected = {}
        for path, keys in zip(paths, map_keys):
            if keys is None:
                continue
            expected[path] = sorted((expected_map[k] for k in keys))
        for path in result:
            result[path].sort()
        self.assertEqual(expected, result)

    def assertBisectRecursive(self, expected_map, map_keys, state, paths):
        """Assert the return value of a recursive bisection.

        :param expected_map: A map from key => entry value
        :param map_keys: A list of paths we expect to be returned.
            Something like ['a', 'b', 'f', 'b/d', 'b/d2']
        :param state: The DirState object.
        :param paths: A list of files and directories. It will be broken up
            into (dir, name) pairs and sorted before calling _bisect_recursive.
        """
        expected = {}
        for key in map_keys:
            entry = expected_map[key]
            dir_name_id, trees_info = entry
            expected[dir_name_id] = trees_info
        result = state._bisect_recursive(paths)
        self.assertEqual(expected, result)

    def test_bisect_each(self):
        """Find a single record using bisect."""
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisect(expected, [[b'']], state, [b''])
        self.assertBisect(expected, [[b'a']], state, [b'a'])
        self.assertBisect(expected, [[b'b']], state, [b'b'])
        self.assertBisect(expected, [[b'b/c']], state, [b'b/c'])
        self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
        self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
        self.assertBisect(expected, [[b'b-c']], state, [b'b-c'])
        self.assertBisect(expected, [[b'f']], state, [b'f'])

    def test_bisect_multi(self):
        """Bisect can be used to find multiple records at the same time."""
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisect(expected, [[b'a'], [b'b'], [b'f']], state, [b'a', b'b', b'f'])
        self.assertBisect(expected, [[b'f'], [b'b/d'], [b'b/d/e']], state, [b'f', b'b/d', b'b/d/e'])
        self.assertBisect(expected, [[b'b'], [b'b-c'], [b'b/c']], state, [b'b', b'b-c', b'b/c'])

    def test_bisect_one_page(self):
        """Test bisect when there is only 1 page to read"""
        tree, state, expected = self.create_basic_dirstate()
        state._bisect_page_size = 5000
        self.assertBisect(expected, [[b'']], state, [b''])
        self.assertBisect(expected, [[b'a']], state, [b'a'])
        self.assertBisect(expected, [[b'b']], state, [b'b'])
        self.assertBisect(expected, [[b'b/c']], state, [b'b/c'])
        self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
        self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
        self.assertBisect(expected, [[b'b-c']], state, [b'b-c'])
        self.assertBisect(expected, [[b'f']], state, [b'f'])
        self.assertBisect(expected, [[b'a'], [b'b'], [b'f']], state, [b'a', b'b', b'f'])
        self.assertBisect(expected, [[b'b/d'], [b'b/d/e'], [b'f']], state, [b'b/d', b'b/d/e', b'f'])
        self.assertBisect(expected, [[b'b'], [b'b/c'], [b'b-c']], state, [b'b', b'b/c', b'b-c'])

    def test_bisect_duplicate_paths(self):
        """When bisecting for a path, handle multiple entries."""
        tree, state, expected = self.create_duplicated_dirstate()
        self.assertBisect(expected, [[b'']], state, [b''])
        self.assertBisect(expected, [[b'a', b'a2']], state, [b'a'])
        self.assertBisect(expected, [[b'b', b'b2']], state, [b'b'])
        self.assertBisect(expected, [[b'b/c', b'b/c2']], state, [b'b/c'])
        self.assertBisect(expected, [[b'b/d', b'b/d2']], state, [b'b/d'])
        self.assertBisect(expected, [[b'b/d/e', b'b/d/e2']], state, [b'b/d/e'])
        self.assertBisect(expected, [[b'b-c', b'b-c2']], state, [b'b-c'])
        self.assertBisect(expected, [[b'f', b'f2']], state, [b'f'])

    def test_bisect_page_size_too_small(self):
        """If the page size is too small, we will auto increase it."""
        tree, state, expected = self.create_basic_dirstate()
        state._bisect_page_size = 50
        self.assertBisect(expected, [None], state, [b'b/e'])
        self.assertBisect(expected, [[b'a']], state, [b'a'])
        self.assertBisect(expected, [[b'b']], state, [b'b'])
        self.assertBisect(expected, [[b'b/c']], state, [b'b/c'])
        self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
        self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
        self.assertBisect(expected, [[b'b-c']], state, [b'b-c'])
        self.assertBisect(expected, [[b'f']], state, [b'f'])

    def test_bisect_missing(self):
        """Test that bisect return None if it cannot find a path."""
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisect(expected, [None], state, [b'foo'])
        self.assertBisect(expected, [None], state, [b'b/foo'])
        self.assertBisect(expected, [None], state, [b'bar/foo'])
        self.assertBisect(expected, [None], state, [b'b-c/foo'])
        self.assertBisect(expected, [[b'a'], None, [b'b/d']], state, [b'a', b'foo', b'b/d'])

    def test_bisect_rename(self):
        """Check that we find a renamed row."""
        tree, state, expected = self.create_renamed_dirstate()
        self.assertBisect(expected, [[b'a']], state, [b'a'])
        self.assertBisect(expected, [[b'b/g']], state, [b'b/g'])
        self.assertBisect(expected, [[b'b/d']], state, [b'b/d'])
        self.assertBisect(expected, [[b'h']], state, [b'h'])
        self.assertBisect(expected, [[b'b/d/e']], state, [b'b/d/e'])
        self.assertBisect(expected, [[b'h/e']], state, [b'h/e'])

    def test_bisect_dirblocks(self):
        tree, state, expected = self.create_duplicated_dirstate()
        self.assertBisectDirBlocks(expected, [[b'', b'a', b'a2', b'b', b'b2', b'b-c', b'b-c2', b'f', b'f2']], state, [b''])
        self.assertBisectDirBlocks(expected, [[b'b/c', b'b/c2', b'b/d', b'b/d2']], state, [b'b'])
        self.assertBisectDirBlocks(expected, [[b'b/d/e', b'b/d/e2']], state, [b'b/d'])
        self.assertBisectDirBlocks(expected, [[b'', b'a', b'a2', b'b', b'b2', b'b-c', b'b-c2', b'f', b'f2'], [b'b/c', b'b/c2', b'b/d', b'b/d2'], [b'b/d/e', b'b/d/e2']], state, [b'', b'b', b'b/d'])

    def test_bisect_dirblocks_missing(self):
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisectDirBlocks(expected, [[b'b/d/e'], None], state, [b'b/d', b'b/e'])
        self.assertBisectDirBlocks(expected, [None], state, [b'a'])
        self.assertBisectDirBlocks(expected, [None], state, [b'b/c'])
        self.assertBisectDirBlocks(expected, [None], state, [b'c'])
        self.assertBisectDirBlocks(expected, [None], state, [b'b/d/e'])
        self.assertBisectDirBlocks(expected, [None], state, [b'f'])

    def test_bisect_recursive_each(self):
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisectRecursive(expected, [b'a'], state, [b'a'])
        self.assertBisectRecursive(expected, [b'b/c'], state, [b'b/c'])
        self.assertBisectRecursive(expected, [b'b/d/e'], state, [b'b/d/e'])
        self.assertBisectRecursive(expected, [b'b-c'], state, [b'b-c'])
        self.assertBisectRecursive(expected, [b'b/d', b'b/d/e'], state, [b'b/d'])
        self.assertBisectRecursive(expected, [b'b', b'b/c', b'b/d', b'b/d/e'], state, [b'b'])
        self.assertBisectRecursive(expected, [b'', b'a', b'b', b'b-c', b'f', b'b/c', b'b/d', b'b/d/e'], state, [b''])

    def test_bisect_recursive_multiple(self):
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisectRecursive(expected, [b'a', b'b/c'], state, [b'a', b'b/c'])
        self.assertBisectRecursive(expected, [b'b/d', b'b/d/e'], state, [b'b/d', b'b/d/e'])

    def test_bisect_recursive_missing(self):
        tree, state, expected = self.create_basic_dirstate()
        self.assertBisectRecursive(expected, [], state, [b'd'])
        self.assertBisectRecursive(expected, [], state, [b'b/e'])
        self.assertBisectRecursive(expected, [], state, [b'g'])
        self.assertBisectRecursive(expected, [b'a'], state, [b'a', b'g'])

    def test_bisect_recursive_renamed(self):
        tree, state, expected = self.create_renamed_dirstate()
        self.assertBisectRecursive(expected, [b'a', b'b/g'], state, [b'a'])
        self.assertBisectRecursive(expected, [b'a', b'b/g'], state, [b'b/g'])
        self.assertBisectRecursive(expected, [b'a', b'b', b'b/c', b'b/d', b'b/d/e', b'b/g', b'h', b'h/e'], state, [b'b'])