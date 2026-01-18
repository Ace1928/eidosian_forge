import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_py_multi(self):
    self.paste('\n        >>> x = [1,2,3]\n        >>> y = []\n        >>> for i in x:\n        ...     y.append(i**2)\n        ... \n        ')
    self.assertEqual(ip.user_ns['x'], [1, 2, 3])
    self.assertEqual(ip.user_ns['y'], [1, 4, 9])