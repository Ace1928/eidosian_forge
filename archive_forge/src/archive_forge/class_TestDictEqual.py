from unittest import TestCase
from jsonschema._utils import equal
class TestDictEqual(TestCase):

    def test_equal_dictionaries(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        dict_2 = {'c': 'd', 'a': 'b'}
        self.assertTrue(equal(dict_1, dict_2))

    def test_missing_key(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        dict_2 = {'c': 'd', 'x': 'b'}
        self.assertFalse(equal(dict_1, dict_2))

    def test_additional_key(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        dict_2 = {'c': 'd', 'a': 'b', 'x': 'x'}
        self.assertFalse(equal(dict_1, dict_2))

    def test_missing_value(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        dict_2 = {'c': 'd', 'a': 'x'}
        self.assertFalse(equal(dict_1, dict_2))

    def test_empty_dictionaries(self):
        dict_1 = {}
        dict_2 = {}
        self.assertTrue(equal(dict_1, dict_2))

    def test_one_none(self):
        dict_1 = None
        dict_2 = {'a': 'b', 'c': 'd'}
        self.assertFalse(equal(dict_1, dict_2))

    def test_same_item(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        self.assertTrue(equal(dict_1, dict_1))

    def test_nested_equal(self):
        dict_1 = {'a': {'a': 'b', 'c': 'd'}, 'c': 'd'}
        dict_2 = {'c': 'd', 'a': {'a': 'b', 'c': 'd'}}
        self.assertTrue(equal(dict_1, dict_2))

    def test_nested_dict_unequal(self):
        dict_1 = {'a': {'a': 'b', 'c': 'd'}, 'c': 'd'}
        dict_2 = {'c': 'd', 'a': {'a': 'b', 'c': 'x'}}
        self.assertFalse(equal(dict_1, dict_2))

    def test_mixed_nested_equal(self):
        dict_1 = {'a': ['a', 'b', 'c', 'd'], 'c': 'd'}
        dict_2 = {'c': 'd', 'a': ['a', 'b', 'c', 'd']}
        self.assertTrue(equal(dict_1, dict_2))

    def test_nested_list_unequal(self):
        dict_1 = {'a': ['a', 'b', 'c', 'd'], 'c': 'd'}
        dict_2 = {'c': 'd', 'a': ['b', 'c', 'd', 'a']}
        self.assertFalse(equal(dict_1, dict_2))