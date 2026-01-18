from unittest import TestCase
from jsonschema._utils import equal
def test_one_none(self):
    list_1 = None
    list_2 = []
    self.assertFalse(equal(list_1, list_2))