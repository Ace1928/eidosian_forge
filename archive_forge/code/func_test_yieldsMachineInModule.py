import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachineInModule(self):
    """
        When given a L{twisted.python.modules.PythonModule} that refers to
        a module that contains a L{MethodicalMachine},
        L{findMachinesViaWrapper} yields that machine and its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        rootMachine = MethodicalMachine()\n        '
    module = self.makeModule(source, self.pathDir, 'root.py')
    rootMachine = self.loadModuleAsDict(module)['root.rootMachine'].load()
    self.assertIn(('root.rootMachine', rootMachine), list(self.findMachinesViaWrapper(module)))