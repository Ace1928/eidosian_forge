import os
import sys
from breezy import branch, osutils, registry, tests
def _iter_them(self, iter_func_name):
    iter_func = getattr(self.registry, iter_func_name, None)
    self.assertIsNot(None, iter_func)
    count = 0
    for name, func in iter_func():
        count += 1
        self.assertFalse(name in ('hidden', 'more hidden'))
        if func is not None:
            func()
    self.assertEqual(4, count)