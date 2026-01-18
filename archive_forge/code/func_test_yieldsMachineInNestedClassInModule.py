import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachineInNestedClassInModule(self):
    """
        When given a L{twisted.python.modules.PythonModule} that refers to
        the original module of a nested class containing a
        L{MethodicalMachine}, L{findMachinesViaWrapper} yields that
        machine and its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        class PythonClass(object):\n            class NestedClass(object):\n                _classMachine = MethodicalMachine()\n        '
    module = self.makeModule(source, self.pathDir, 'nestedcls.py')
    PythonClass = self.loadModuleAsDict(module)['nestedcls.PythonClass'].load()
    self.assertIn(('nestedcls.PythonClass.NestedClass._classMachine', PythonClass.NestedClass._classMachine), list(self.findMachinesViaWrapper(module)))