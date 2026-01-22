import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
class TestStats(tests.TestCaseInTempDir):
    _test_needs_features = [features.lsprof_feature]

    def setUp(self):
        super(tests.TestCaseInTempDir, self).setUp()
        self.stats = _collect_stats()

    def _temppath(self, ext):
        return osutils.pathjoin(self.test_dir, 'tmp_profile_data.' + ext)

    def test_save_to_txt(self):
        path = self._temppath('txt')
        self.stats.save(path)
        with open(path) as f:
            lines = f.readlines()
            self.assertEqual(lines[0], _TXT_HEADER)

    def test_save_to_callgrind(self):
        path1 = self._temppath('callgrind')
        self.stats.save(path1)
        with open(path1) as f:
            self.assertEqual(f.readline(), 'events: Ticks\n')
        path2 = osutils.pathjoin(self.test_dir, 'callgrind.out.foo')
        self.stats.save(path2)
        with open(path2) as f:
            self.assertEqual(f.readline(), 'events: Ticks\n')
        path3 = self._temppath('txt')
        self.stats.save(path3, format='callgrind')
        with open(path3) as f:
            self.assertEqual(f.readline(), 'events: Ticks\n')

    def test_save_to_pickle(self):
        path = self._temppath('pkl')
        self.stats.save(path)
        with open(path, 'rb') as f:
            data1 = pickle.load(f)
            self.assertEqual(type(data1), lsprof.Stats)

    def test_sort(self):
        self.stats.sort()
        code_list = [d.inlinetime for d in self.stats.data]
        self.assertEqual(code_list, sorted(code_list, reverse=True))

    def test_sort_totaltime(self):
        self.stats.sort('totaltime')
        code_list = [d.totaltime for d in self.stats.data]
        self.assertEqual(code_list, sorted(code_list, reverse=True))

    def test_sort_code(self):
        """Cannot sort by code object would need to get filename etc."""
        self.assertRaises(ValueError, self.stats.sort, 'code')