import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
class TestWeakIDDict(unittest.TestCase):

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

    def test_weak_keys_values(self):
        wd = WeakIDDict()
        keep = []
        dont_keep = []
        values = list(map(WeakreffableInt, range(10)))
        for n, i in enumerate(values, 1):
            key = AllTheSame()
            if not i.value % 2:
                keep.append(key)
            else:
                dont_keep.append(key)
            wd[key] = i
            del key
            self.assertEqual(len(wd), n)
        self.assertEqual(len(wd), 10)
        del dont_keep
        self.assertEqual(len(wd), 5)
        self.assertCountEqual(list(wd.values()), list(map(WeakreffableInt, [0, 2, 4, 6, 8])))
        self.assertEqual([wd[k] for k in keep], list(map(WeakreffableInt, [0, 2, 4, 6, 8])))
        self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in wd])
        self.assertCountEqual([id(k) for k in wd.keys()], [id(k) for k in keep])
        del values[0:2]
        self.assertEqual(len(wd), 4)
        del values[0:2]
        self.assertEqual(len(wd), 3)
        del values[0:2]
        self.assertEqual(len(wd), 2)
        del values[0:2]
        self.assertEqual(len(wd), 1)
        del values[0:2]
        self.assertEqual(len(wd), 0)

    def test_weak_id_dict_str_representation(self):
        """ test string representation of the WeakIDDict class. """
        weak_id_dict = WeakIDDict()
        desired_repr = '<WeakIDDict at 0x{0:x}>'.format(id(weak_id_dict))
        self.assertEqual(desired_repr, str(weak_id_dict))
        self.assertEqual(desired_repr, repr(weak_id_dict))

    def test_weak_id_key_dict_str_representation(self):
        """ test string representation of the WeakIDKeyDict class. """
        weak_id_key_dict = WeakIDKeyDict()
        desired_repr = f'<WeakIDKeyDict at 0x{id(weak_id_key_dict):x}>'
        self.assertEqual(desired_repr, str(weak_id_key_dict))
        self.assertEqual(desired_repr, repr(weak_id_key_dict))