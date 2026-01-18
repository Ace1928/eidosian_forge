from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_on_noarg(self):

    def test():
        pass
    self.assertEqual(self.signature(test), ((), Ellipsis))