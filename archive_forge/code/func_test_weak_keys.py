import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
def test_weak_keys(self):
    wd = WeakIDKeyDict()
    keep = []
    dont_keep = []
    values = list(range(10))
    for n, i in enumerate(values, 1):
        key = AllTheSame()
        if not i % 2:
            keep.append(key)
        else:
            dont_keep.append(key)
        wd[key] = i
        del key
        self.assertEqual(len(wd), n)
    self.assertEqual(len(wd), 10)
    del dont_keep
    self.assertEqual(len(wd), 5)
    self.assertCountEqual(list(wd.values()), list(range(0, 10, 2)))
    self.assertEqual([wd[k] for k in keep], list(range(0, 10, 2)))
    self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in wd])
    self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in keep])