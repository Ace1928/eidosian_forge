import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def test_syntaxerror(self):
    i, a = self.interp_errlog()
    i.runsource('1.1.1.1')
    if (3, 10, 1) <= sys.version_info[:3]:
        expected = '  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + '\n    1.1.1.1\n       ^^\n' + bold(red('SyntaxError')) + ': ' + cyan('invalid syntax') + '\n'
    elif (3, 10) <= sys.version_info[:2]:
        expected = '  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + '\n    1.1.1.1\n    ^^^^^\n' + bold(red('SyntaxError')) + ': ' + cyan('invalid syntax. Perhaps you forgot a comma?') + '\n'
    elif (3, 8) <= sys.version_info[:2]:
        expected = '  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + '\n    1.1.1.1\n       ^\n' + bold(red('SyntaxError')) + ': ' + cyan('invalid syntax') + '\n'
    elif pypy:
        expected = '  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + '\n    1.1.1.1\n       ^\n' + bold(red('SyntaxError')) + ': ' + cyan('invalid syntax') + '\n'
    else:
        expected = '  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + '\n    1.1.1.1\n        ^\n' + bold(red('SyntaxError')) + ': ' + cyan('invalid syntax') + '\n'
    self.assertMultiLineEqual(str(plain('').join(a)), str(expected))
    self.assertEqual(plain('').join(a), expected)