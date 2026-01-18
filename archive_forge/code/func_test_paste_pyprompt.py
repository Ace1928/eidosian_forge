import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_pyprompt(self):
    ip.user_ns.pop('x', None)
    self.paste('>>> x=2')
    self.assertEqual(ip.user_ns['x'], 2)
    ip.user_ns.pop('x')