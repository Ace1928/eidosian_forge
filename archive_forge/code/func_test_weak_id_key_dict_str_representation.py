import unittest
from traits.util.weakiddict import WeakIDDict, WeakIDKeyDict
def test_weak_id_key_dict_str_representation(self):
    """ test string representation of the WeakIDKeyDict class. """
    weak_id_key_dict = WeakIDKeyDict()
    desired_repr = f'<WeakIDKeyDict at 0x{id(weak_id_key_dict):x}>'
    self.assertEqual(desired_repr, str(weak_id_key_dict))
    self.assertEqual(desired_repr, repr(weak_id_key_dict))