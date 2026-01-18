import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_email2(self):
    """Email again; some programs add a space also at each quoting level"""
    self.paste('        > > def foo(x):\n        > >     return x + 1\n        > > yy = foo(2.1)     ')
    self.assertEqual(ip.user_ns['yy'], 3.1)