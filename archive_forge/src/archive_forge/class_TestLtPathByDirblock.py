import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestLtPathByDirblock(tests.TestCase):
    """Test an implementation of _lt_path_by_dirblock()

    _lt_path_by_dirblock() compares two paths using the sort order used by
    DirState. All paths in the same directory are sorted together.

    Child test cases can override ``get_lt_path_by_dirblock`` to test a specific
    implementation.
    """

    def get_lt_path_by_dirblock(self):
        """Get a specific implementation of _lt_path_by_dirblock."""
        from .._dirstate_helpers_py import _lt_path_by_dirblock
        return _lt_path_by_dirblock

    def assertLtPathByDirblock(self, paths):
        """Compare all paths and make sure they evaluate to the correct order.

        This does N^2 comparisons. It is assumed that ``paths`` is properly
        sorted list.

        :param paths: a sorted list of paths to compare
        """

        def _key(p):
            dirname, basename = os.path.split(p)
            return (dirname.split(b'/'), basename)
        self.assertEqual(sorted(paths, key=_key), paths)
        lt_path_by_dirblock = self.get_lt_path_by_dirblock()
        for idx1, path1 in enumerate(paths):
            for idx2, path2 in enumerate(paths):
                lt_result = lt_path_by_dirblock(path1, path2)
                self.assertEqual(idx1 < idx2, lt_result, '%s did not state that %r < %r, lt=%s' % (lt_path_by_dirblock.__name__, path1, path2, lt_result))

    def test_cmp_simple_paths(self):
        """Compare against the empty string."""
        self.assertLtPathByDirblock([b'', b'a', b'ab', b'abc', b'a/b/c', b'b/d/e'])
        self.assertLtPathByDirblock([b'kl', b'ab/cd', b'ab/ef', b'gh/ij'])

    def test_tricky_paths(self):
        self.assertLtPathByDirblock([b'', b'a', b'a-a', b'a=a', b'b', b'a/a', b'a/a-a', b'a/a=a', b'a/b', b'a/a/a', b'a/a/a-a', b'a/a/a=a', b'a/a/a/a', b'a/a/a/b', b'a/a/a-a/a', b'a/a/a-a/b', b'a/a/a=a/a', b'a/a/a=a/b', b'a/a-a/a', b'a/a-a/a/a', b'a/a-a/a/b', b'a/a=a/a', b'a/b/a', b'a/b/b', b'a-a/a', b'a-a/b', b'a=a/a', b'a=a/b', b'b/a', b'b/b'])
        self.assertLtPathByDirblock([b'', b'a', b'a-a', b'a-z', b'a=a', b'a=z', b'a/a', b'a/a-a', b'a/a-z', b'a/a=a', b'a/a=z', b'a/z', b'a/z-a', b'a/z-z', b'a/z=a', b'a/z=z', b'a/a/a', b'a/a/z', b'a/a-a/a', b'a/a-z/z', b'a/a=a/a', b'a/a=z/z', b'a/z/a', b'a/z/z', b'a-a/a', b'a-z/z', b'a=a/a', b'a=z/z'])

    def test_unicode_not_allowed(self):
        lt_path_by_dirblock = self.get_lt_path_by_dirblock()
        self.assertRaises(TypeError, lt_path_by_dirblock, 'Uni', 'str')
        self.assertRaises(TypeError, lt_path_by_dirblock, 'str', 'Uni')
        self.assertRaises(TypeError, lt_path_by_dirblock, 'Uni', 'Uni')
        self.assertRaises(TypeError, lt_path_by_dirblock, 'x/Uni', 'x/str')
        self.assertRaises(TypeError, lt_path_by_dirblock, 'x/str', 'x/Uni')
        self.assertRaises(TypeError, lt_path_by_dirblock, 'x/Uni', 'x/Uni')

    def test_nonascii(self):
        self.assertLtPathByDirblock([b'', b'a', b'\xc2\xb5', b'\xc3\xa5', b'a/a', b'a/\xc2\xb5', b'a/\xc3\xa5', b'a/a/a', b'a/a/\xc2\xb5', b'a/a/\xc3\xa5', b'a/\xc2\xb5/a', b'a/\xc2\xb5/\xc2\xb5', b'a/\xc2\xb5/\xc3\xa5', b'a/\xc3\xa5/a', b'a/\xc3\xa5/\xc2\xb5', b'a/\xc3\xa5/\xc3\xa5', b'\xc2\xb5/a', b'\xc2\xb5/\xc2\xb5', b'\xc2\xb5/\xc3\xa5', b'\xc3\xa5/a', b'\xc3\xa5/\xc2\xb5', b'\xc3\xa5/\xc3\xa5'])