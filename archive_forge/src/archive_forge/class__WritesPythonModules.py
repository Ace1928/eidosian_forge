import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
class _WritesPythonModules(TestCase):
    """
    A helper that enables generating Python module test fixtures.
    """

    def setUp(self):
        super(_WritesPythonModules, self).setUp()
        from twisted.python.modules import getModule, PythonPath
        from twisted.python.filepath import FilePath
        self.getModule = getModule
        self.PythonPath = PythonPath
        self.FilePath = FilePath
        self.originalSysModules = set(sys.modules.keys())
        self.savedSysPath = sys.path[:]
        self.pathDir = tempfile.mkdtemp()
        self.makeImportable(self.pathDir)

    def tearDown(self):
        super(_WritesPythonModules, self).tearDown()
        sys.path[:] = self.savedSysPath
        modulesToDelete = sys.modules.keys() - self.originalSysModules
        for module in modulesToDelete:
            del sys.modules[module]
        shutil.rmtree(self.pathDir)

    def makeImportable(self, path):
        sys.path.append(path)

    def writeSourceInto(self, source, path, moduleName):
        directory = self.FilePath(path)
        module = directory.child(moduleName)
        with open(module.path, 'w') as f:
            f.write(textwrap.dedent(source))
        return self.PythonPath([directory.path])

    def makeModule(self, source, path, moduleName):
        pythonModuleName, _ = os.path.splitext(moduleName)
        return self.writeSourceInto(source, path, moduleName)[pythonModuleName]

    def attributesAsDict(self, hasIterAttributes):
        return {attr.name: attr for attr in hasIterAttributes.iterAttributes()}

    def loadModuleAsDict(self, module):
        module.load()
        return self.attributesAsDict(module)

    def makeModuleAsDict(self, source, path, name):
        return self.loadModuleAsDict(self.makeModule(source, path, name))