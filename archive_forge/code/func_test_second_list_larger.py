from unittest import TestCase
from jsonschema._utils import equal
def test_second_list_larger(self):
    list_1 = ['a', 'b']
    list_2 = ['a', 'b', 'c']
    self.assertFalse(equal(list_1, list_2))