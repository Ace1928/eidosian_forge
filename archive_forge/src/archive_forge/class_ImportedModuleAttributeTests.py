import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
class ImportedModuleAttributeTests(TwistedModulesMixin, SynchronousTestCase):
    """
    Tests for L{deprecatedModuleAttribute} which involve loading a module via
    'import'.
    """
    _packageInit = "from twisted.python.deprecate import deprecatedModuleAttribute\nfrom incremental import Version\n\ndeprecatedModuleAttribute(\n    Version('Package', 1, 2, 3), 'message', __name__, 'module')\n"

    def pathEntryTree(self, tree):
        """
        Create some files in a hierarchy, based on a dictionary describing those
        files.  The resulting hierarchy will be placed onto sys.path for the
        duration of the test.

        @param tree: A dictionary representing a directory structure.  Keys are
            strings, representing filenames, dictionary values represent
            directories, string values represent file contents.

        @return: another dictionary similar to the input, with file content
            strings replaced with L{FilePath} objects pointing at where those
            contents are now stored.
        """

        def makeSomeFiles(pathobj, dirdict):
            pathdict = {}
            for key, value in dirdict.items():
                child = pathobj.child(key)
                if isinstance(value, bytes):
                    pathdict[key] = child
                    child.setContent(value)
                elif isinstance(value, dict):
                    child.createDirectory()
                    pathdict[key] = makeSomeFiles(child, value)
                else:
                    raise ValueError('only strings and dicts allowed as values')
            return pathdict
        base = FilePath(self.mktemp().encode('utf-8'))
        base.makedirs()
        result = makeSomeFiles(base, tree)
        self.replaceSysPath([base.path.decode('utf-8')] + sys.path)
        self.replaceSysModules(sys.modules.copy())
        return result

    def simpleModuleEntry(self):
        """
        Add a sample module and package to the path, returning a L{FilePath}
        pointing at the module which will be loadable as C{package.module}.
        """
        paths = self.pathEntryTree({b'package': {b'__init__.py': self._packageInit.encode('utf-8'), b'module.py': b''}})
        return paths[b'package'][b'module.py']

    def checkOneWarning(self, modulePath):
        """
        Verification logic for L{test_deprecatedModule}.
        """
        from package import module
        self.assertEqual(FilePath(module.__file__.encode('utf-8')), modulePath)
        emitted = self.flushWarnings([self.checkOneWarning])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]['message'], 'package.module was deprecated in Package 1.2.3: message')
        self.assertEqual(emitted[0]['category'], DeprecationWarning)

    def test_deprecatedModule(self):
        """
        If L{deprecatedModuleAttribute} is used to deprecate a module attribute
        of a package, only one deprecation warning is emitted when the
        deprecated module is imported.
        """
        self.checkOneWarning(self.simpleModuleEntry())

    def test_deprecatedModuleMultipleTimes(self):
        """
        If L{deprecatedModuleAttribute} is used to deprecate a module attribute
        of a package, only one deprecation warning is emitted when the
        deprecated module is subsequently imported.
        """
        mp = self.simpleModuleEntry()
        self.checkOneWarning(mp)
        self.checkOneWarning(mp)
        for x in range(2):
            self.checkOneWarning(mp)