import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class EscapeTests(unittest.TestCase):

    def test_escape(self):
        """
        Verify _shellcomp.escape() function
        """
        esc = _shellcomp.escape
        test = '$'
        self.assertEqual(esc(test), "'$'")
        test = 'A--\'$"\\`--B'
        self.assertEqual(esc(test), '"A--\'\\$\\"\\\\\\`--B"')