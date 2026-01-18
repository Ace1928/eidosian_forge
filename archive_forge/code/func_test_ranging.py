import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
def test_ranging(self):
    self.assertEqual(intranges_from_list(list(range(293, 499)) + list(range(4888, 9876))), (_encode_range(293, 499), _encode_range(4888, 9876)))