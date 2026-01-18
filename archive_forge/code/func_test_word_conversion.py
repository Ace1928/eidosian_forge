import unittest
from traits.util.camel_case import camel_case_to_python, camel_case_to_words
def test_word_conversion(self):
    """ Does CamelCase -> words work?
        """
    self.assertEqual(camel_case_to_words('FooBarBaz'), 'Foo Bar Baz')