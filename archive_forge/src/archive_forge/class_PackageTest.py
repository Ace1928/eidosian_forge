import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
class PackageTest(unittest.SynchronousTestCase):
    files = [('badpackage/__init__.py', 'frotz\n'), ('badpackage/test_module.py', ''), ('unimportablepackage/__init__.py', ''), ('unimportablepackage/test_module.py', 'import notarealmoduleok\n'), ('package2/__init__.py', ''), ('package2/test_module.py', 'import frotz\n'), ('package/__init__.py', ''), ('package/frotz.py', 'frotz\n'), ('package/test_bad_module.py', 'raise ZeroDivisionError("fake error")'), ('package/test_dos_module.py', dosModule), ('package/test_import_module.py', 'import frotz'), ('package/test_module.py', testModule), ('goodpackage/__init__.py', ''), ('goodpackage/test_sample.py', testSample), ('goodpackage/sub/__init__.py', ''), ('goodpackage/sub/test_sample.py', testSample), ('inheritancepackage/__init__.py', ''), ('inheritancepackage/test_x.py', testInheritanceSample)]

    def _toModuleName(self, filename):
        name = os.path.splitext(filename)[0]
        segs = name.split('/')
        if segs[-1] == '__init__':
            segs = segs[:-1]
        return '.'.join(segs)

    def getModules(self):
        """
        Return matching module names for files listed in C{self.files}.
        """
        return [self._toModuleName(filename) for filename, code in self.files]

    def cleanUpModules(self):
        modules = self.getModules()
        modules.sort()
        modules.reverse()
        for module in modules:
            try:
                del sys.modules[module]
            except KeyError:
                pass

    def createFiles(self, files, parentDir='.'):
        for filename, contents in self.files:
            filename = os.path.join(parentDir, filename)
            self._createDirectory(filename)
            with open(filename, 'w') as fd:
                fd.write(contents)

    def _createDirectory(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def setUp(self, parentDir=None):
        invalidateImportCaches()
        if parentDir is None:
            parentDir = self.mktemp()
        self.parent = parentDir
        self.createFiles(self.files, parentDir)

    def tearDown(self):
        self.cleanUpModules()