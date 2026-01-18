import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachineInClassInModule(self):
    """
        When given a L{twisted.python.modules.PythonModule} that refers to
        the original module of a class containing a
        L{MethodicalMachine}, L{findMachinesViaWrapper} yields that
        machine and its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        class PythonClass(object):\n            _classMachine = MethodicalMachine()\n        '
    module = self.makeModule(source, self.pathDir, 'clsmod.py')
    PythonClass = self.loadModuleAsDict(module)['clsmod.PythonClass'].load()
    self.assertIn(('clsmod.PythonClass._classMachine', PythonClass._classMachine), list(self.findMachinesViaWrapper(module)))