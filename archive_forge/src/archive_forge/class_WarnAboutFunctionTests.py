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
class WarnAboutFunctionTests(SynchronousTestCase):
    """
    Tests for L{twisted.python.deprecate.warnAboutFunction} which allows the
    callers of a function to issue a C{DeprecationWarning} about that function.
    """

    def setUp(self):
        """
        Create a file that will have known line numbers when emitting warnings.
        """
        self.package = FilePath(self.mktemp()).child('twisted_private_helper')
        self.package.makedirs()
        self.package.child('__init__.py').setContent(b'')
        self.package.child('module.py').setContent(b'\n"A module string"\n\nfrom twisted.python import deprecate\n\ndef testFunction():\n    "A doc string"\n    a = 1 + 2\n    return a\n\ndef callTestFunction():\n    b = testFunction()\n    if b == 3:\n        deprecate.warnAboutFunction(testFunction, "A Warning String")\n')
        self.package.child('pep626.py').setContent(b'\n"A module string"\n\nfrom twisted.python import deprecate\n\ndef noop():\n    pass\n\ndef testFunction(a=1, b=1):\n    "A doc string"\n    if a:\n        if b:\n            noop()\n        else:\n            pass\n\ndef callTestFunction():\n    b = testFunction()\n    if b is None:\n        deprecate.warnAboutFunction(testFunction, "A Warning String")\n')
        packagePath = self.package.parent().path
        sys.path.insert(0, packagePath)
        self.addCleanup(sys.path.remove, packagePath)
        modules = sys.modules.copy()
        self.addCleanup(lambda: (sys.modules.clear(), sys.modules.update(modules)))
        if platform.isWindows():
            self.flushWarnings()

    def test_warning(self):
        """
        L{deprecate.warnAboutFunction} emits a warning the file and line number
        of which point to the beginning of the implementation of the function
        passed to it.
        """

        def aFunc():
            pass
        deprecate.warnAboutFunction(aFunc, 'A Warning Message')
        warningsShown = self.flushWarnings()
        filename = __file__
        if filename.lower().endswith('.pyc'):
            filename = filename[:-1]
        self.assertSamePath(FilePath(warningsShown[0]['filename']), FilePath(filename))
        self.assertEqual(warningsShown[0]['message'], 'A Warning Message')

    def test_warningLineNumber(self):
        """
        L{deprecate.warnAboutFunction} emits a C{DeprecationWarning} with the
        number of a line within the implementation of the function passed to it.
        """
        from twisted_private_helper import module
        module.callTestFunction()
        warningsShown = self.flushWarnings()
        self.assertSamePath(FilePath(warningsShown[0]['filename'].encode('utf-8')), self.package.sibling(b'twisted_private_helper').child(b'module.py'))
        self.assertEqual(warningsShown[0]['lineno'], 9)
        self.assertEqual(warningsShown[0]['message'], 'A Warning String')
        self.assertEqual(len(warningsShown), 1)

    def test_warningLineNumberDisFindlinestarts(self):
        """
        L{deprecate.warnAboutFunction} emits a C{DeprecationWarning} with the
        number of a line within the implementation handling the case in which
        dis.findlinestarts returns the lines in random order.
        """
        from twisted_private_helper import pep626
        pep626.callTestFunction()
        warningsShown = self.flushWarnings()
        self.assertSamePath(FilePath(warningsShown[0]['filename'].encode('utf-8')), self.package.sibling(b'twisted_private_helper').child(b'pep626.py'))
        self.assertEqual(warningsShown[0]['lineno'], 15)
        self.assertEqual(warningsShown[0]['message'], 'A Warning String')
        self.assertEqual(len(warningsShown), 1)

    def assertSamePath(self, first, second):
        """
        Assert that the two paths are the same, considering case normalization
        appropriate for the current platform.

        @type first: L{FilePath}
        @type second: L{FilePath}

        @raise C{self.failureType}: If the paths are not the same.
        """
        self.assertTrue(normcase(first.path) == normcase(second.path), f'{first!r} != {second!r}')

    def test_renamedFile(self):
        """
        Even if the implementation of a deprecated function is moved around on
        the filesystem, the line number in the warning emitted by
        L{deprecate.warnAboutFunction} points to a line in the implementation of
        the deprecated function.
        """
        from twisted_private_helper import module
        del sys.modules['twisted_private_helper']
        del sys.modules[module.__name__]
        self.package.moveTo(self.package.sibling(b'twisted_renamed_helper'))
        if invalidate_caches:
            invalidate_caches()
        from twisted_renamed_helper import module
        self.addCleanup(sys.modules.pop, 'twisted_renamed_helper')
        self.addCleanup(sys.modules.pop, module.__name__)
        module.callTestFunction()
        warningsShown = self.flushWarnings([module.testFunction])
        warnedPath = FilePath(warningsShown[0]['filename'].encode('utf-8'))
        expectedPath = self.package.sibling(b'twisted_renamed_helper').child(b'module.py')
        self.assertSamePath(warnedPath, expectedPath)
        self.assertEqual(warningsShown[0]['lineno'], 9)
        self.assertEqual(warningsShown[0]['message'], 'A Warning String')
        self.assertEqual(len(warningsShown), 1)

    def test_filteredWarning(self):
        """
        L{deprecate.warnAboutFunction} emits a warning that will be filtered if
        L{warnings.filterwarning} is called with the module name of the
        deprecated function.
        """
        del warnings.filters[:]
        warnings.filterwarnings(action='ignore', module='twisted_private_helper')
        from twisted_private_helper import module
        module.callTestFunction()
        warningsShown = self.flushWarnings()
        self.assertEqual(len(warningsShown), 0)

    def test_filteredOnceWarning(self):
        """
        L{deprecate.warnAboutFunction} emits a warning that will be filtered
        once if L{warnings.filterwarning} is called with the module name of the
        deprecated function and an action of once.
        """
        del warnings.filters[:]
        warnings.filterwarnings(action='module', module='twisted_private_helper')
        from twisted_private_helper import module
        module.callTestFunction()
        module.callTestFunction()
        warningsShown = self.flushWarnings()
        self.assertEqual(len(warningsShown), 1)
        message = warningsShown[0]['message']
        category = warningsShown[0]['category']
        filename = warningsShown[0]['filename']
        lineno = warningsShown[0]['lineno']
        msg = warnings.formatwarning(message, category, filename, lineno)
        self.assertTrue(msg.endswith('module.py:9: DeprecationWarning: A Warning String\n  return a\n'), f'Unexpected warning string: {msg!r}')