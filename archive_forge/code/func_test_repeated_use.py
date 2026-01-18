import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
def test_repeated_use(self):
    inline_mulcode = 'def f(int a, int b): return a * b'
    self.assertEqual(inline(inline_mulcode)['f'](5, 2), 10)
    self.assertEqual(inline(inline_mulcode)['f'](5, 3), 15)
    self.assertEqual(inline(inline_mulcode)['f'](6, 2), 12)
    self.assertEqual(inline(inline_mulcode)['f'](5, 2), 10)
    f = inline(inline_mulcode)['f']
    self.assertEqual(f(5, 2), 10)
    self.assertEqual(f(5, 3), 15)