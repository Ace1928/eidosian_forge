from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
def test_unknown_types(self):
    orig = [LikeFloat(4), Comparable('hello'), LikeFloat(1), 2.0, Comparable('world'), ToStr(1), NoStr('bogus'), ToStr('a'), ToStr('A'), 3]
    ref = [orig[i] for i in (1, 4, 6, 5, 8, 7, 2, 3, 9, 0)]
    ans = sorted_robust(orig)
    self.assertEqual(len(orig), len(ans))
    for _r, _a in zip(ref, ans):
        self.assertIs(_r, _a)
    self.assertEqual(_robust_sort_keyfcn._typemap[LikeFloat], (1, float.__name__))
    self.assertEqual(_robust_sort_keyfcn._typemap[Comparable], (1, Comparable.__name__))
    self.assertEqual(_robust_sort_keyfcn._typemap[ToStr], (2, ToStr.__name__))
    self.assertEqual(_robust_sort_keyfcn._typemap[NoStr], (3, NoStr.__name__))