import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_yieldsMachine(self):
    """
        When given a L{twisted.python.modules.PythonAttribute} that refers
        directly to a L{MethodicalMachine}, L{findMachinesViaWrapper}
        yields that machine and its FQPN.
        """
    source = '        from automat import MethodicalMachine\n\n        rootMachine = MethodicalMachine()\n        '
    moduleDict = self.makeModuleAsDict(source, self.pathDir, 'root.py')
    rootMachine = moduleDict['root.rootMachine']
    self.assertIn(('root.rootMachine', rootMachine.load()), list(self.findMachinesViaWrapper(rootMachine)))