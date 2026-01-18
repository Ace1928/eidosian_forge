from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_positional_only(self):
    p = inspect.Parameter(None, kind=inspect.Parameter.POSITIONAL_ONLY)
    self.assertEqual(str(p), '<>')
    p = p.replace(name='1')
    self.assertEqual(str(p), '<1>')