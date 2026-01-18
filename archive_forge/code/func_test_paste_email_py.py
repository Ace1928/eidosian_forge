import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_email_py(self):
    """Email quoting of interactive input"""
    self.paste('        >> >>> def f(x):\n        >> ...   return x+1\n        >> ... \n        >> >>> zz = f(2.5)      ')
    self.assertEqual(ip.user_ns['zz'], 3.5)