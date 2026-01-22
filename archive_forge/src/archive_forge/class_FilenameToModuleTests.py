import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class FilenameToModuleTests(TestCase):
    """
    Test L{filenameToModuleName} detection.
    """

    def setUp(self):
        self.path = os.path.join(self.mktemp(), 'fakepackage', 'test')
        os.makedirs(self.path)
        with open(os.path.join(self.path, '__init__.py'), 'w') as f:
            f.write('')
        with open(os.path.join(os.path.dirname(self.path), '__init__.py'), 'w') as f:
            f.write('')

    def test_directory(self):
        """
        L{filenameToModuleName} returns the correct module (a package) given a
        directory.
        """
        module = reflect.filenameToModuleName(self.path)
        self.assertEqual(module, 'fakepackage.test')
        module = reflect.filenameToModuleName(self.path + os.path.sep)
        self.assertEqual(module, 'fakepackage.test')

    def test_file(self):
        """
        L{filenameToModuleName} returns the correct module given the path to
        its file.
        """
        module = reflect.filenameToModuleName(os.path.join(self.path, 'test_reflect.py'))
        self.assertEqual(module, 'fakepackage.test.test_reflect')

    def test_bytes(self):
        """
        L{filenameToModuleName} returns the correct module given a C{bytes}
        path to its file.
        """
        module = reflect.filenameToModuleName(os.path.join(self.path.encode('utf-8'), b'test_reflect.py'))
        self.assertEqual(module, 'fakepackage.test.test_reflect')