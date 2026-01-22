import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class FindMachinesIntegrationTests(_WritesPythonModules):
    """
    Integration tests to check that L{findMachines} yields all
    machines discoverable at or below an FQPN.
    """
    SOURCE = '\n    from automat import MethodicalMachine\n\n    class PythonClass(object):\n        _machine = MethodicalMachine()\n        ignored = "i am ignored"\n\n    rootLevel = MethodicalMachine()\n\n    ignored = "i am ignored"\n    '

    def setUp(self):
        super(FindMachinesIntegrationTests, self).setUp()
        from .._discover import findMachines
        self.findMachines = findMachines
        packageDir = self.FilePath(self.pathDir).child('test_package')
        packageDir.makedirs()
        self.pythonPath = self.PythonPath([self.pathDir])
        self.writeSourceInto(self.SOURCE, packageDir.path, '__init__.py')
        subPackageDir = packageDir.child('subpackage')
        subPackageDir.makedirs()
        subPackageDir.child('__init__.py').touch()
        self.makeModule(self.SOURCE, subPackageDir.path, 'module.py')
        self.packageDict = self.loadModuleAsDict(self.pythonPath['test_package'])
        self.moduleDict = self.loadModuleAsDict(self.pythonPath['test_package']['subpackage']['module'])

    def test_discoverAll(self):
        """
        Given a top-level package FQPN, L{findMachines} discovers all
        L{MethodicalMachine} instances in and below it.
        """
        machines = sorted(self.findMachines('test_package'), key=operator.itemgetter(0))
        tpRootLevel = self.packageDict['test_package.rootLevel'].load()
        tpPythonClass = self.packageDict['test_package.PythonClass'].load()
        mRLAttr = self.moduleDict['test_package.subpackage.module.rootLevel']
        mRootLevel = mRLAttr.load()
        mPCAttr = self.moduleDict['test_package.subpackage.module.PythonClass']
        mPythonClass = mPCAttr.load()
        expectedMachines = sorted([('test_package.rootLevel', tpRootLevel), ('test_package.PythonClass._machine', tpPythonClass._machine), ('test_package.subpackage.module.rootLevel', mRootLevel), ('test_package.subpackage.module.PythonClass._machine', mPythonClass._machine)], key=operator.itemgetter(0))
        self.assertEqual(expectedMachines, machines)