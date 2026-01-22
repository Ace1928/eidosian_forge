import unittest
from Cython.Compiler import Code, UtilityCode
class TestUtilityLoader(unittest.TestCase):
    """
    Test loading UtilityCodes
    """
    expected = ('test {{loader}} prototype', 'test {{loader}} impl')
    required = ('req {{loader}} proto', 'req {{loader}} impl')
    context = dict(loader='Loader')
    name = 'TestUtilityLoader'
    filename = 'TestUtilityLoader.c'
    cls = Code.UtilityCode

    def test_load_as_string(self):
        got = strip_2tup(self.cls.load_as_string(self.name, self.filename))
        self.assertEqual(got, self.expected)

    def test_load(self):
        utility = self.cls.load(self.name, from_file=self.filename)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected)
        required, = utility.requires
        got = strip_2tup((required.proto, required.impl))
        self.assertEqual(got, self.required)
        utility = self.cls.load_cached(self.name, from_file=self.filename)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected)