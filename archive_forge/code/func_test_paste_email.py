import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def test_paste_email(self):
    """Test pasting of email-quoted contents"""
    self.paste('        >> def foo(x):\n        >>     return x + 1\n        >> xx = foo(1.1)')
    self.assertEqual(ip.user_ns['xx'], 2.1)