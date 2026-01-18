import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def test_traceback(self):
    i, a = self.interp_errlog()

    def f():
        return 1 / 0

    def gfunc():
        return f()
    i.runsource('gfunc()')
    global_not_found = "name 'gfunc' is not defined"
    if (3, 11) <= sys.version_info[:2]:
        expected = 'Traceback (most recent call last):\n  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + ', in ' + cyan('<module>') + '\n    gfunc()' + '\n     ^^^^^\n' + bold(red('NameError')) + ': ' + cyan(global_not_found) + '\n'
    else:
        expected = 'Traceback (most recent call last):\n  File ' + green('"<input>"') + ', line ' + bold(magenta('1')) + ', in ' + cyan('<module>') + '\n    gfunc()\n' + bold(red('NameError')) + ': ' + cyan(global_not_found) + '\n'
    self.assertMultiLineEqual(str(plain('').join(a)), str(expected))
    self.assertEqual(plain('').join(a), expected)