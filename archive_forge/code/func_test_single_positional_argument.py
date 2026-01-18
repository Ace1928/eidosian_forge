import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_single_positional_argument(self):

    def test(a):
        pass
    self.assertEqual(self.signature(test), ((('a', Ellipsis, Ellipsis, 'positional_or_keyword'),), Ellipsis))