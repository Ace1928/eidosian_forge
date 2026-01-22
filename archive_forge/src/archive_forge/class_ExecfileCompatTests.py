import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class ExecfileCompatTests(SynchronousTestCase):
    """
    Tests for the Python 3-friendly L{execfile} implementation.
    """

    def writeScript(self, content):
        """
        Write L{content} to a new temporary file, returning the L{FilePath}
        for the new file.
        """
        path = self.mktemp()
        with open(path, 'wb') as f:
            f.write(content.encode('ascii'))
        return FilePath(path.encode('utf-8'))

    def test_execfileGlobals(self):
        """
        L{execfile} executes the specified file in the given global namespace.
        """
        script = self.writeScript('foo += 1\n')
        globalNamespace = {'foo': 1}
        execfile(script.path, globalNamespace)
        self.assertEqual(2, globalNamespace['foo'])

    def test_execfileGlobalsAndLocals(self):
        """
        L{execfile} executes the specified file in the given global and local
        namespaces.
        """
        script = self.writeScript('foo += 1\n')
        globalNamespace = {'foo': 10}
        localNamespace = {'foo': 20}
        execfile(script.path, globalNamespace, localNamespace)
        self.assertEqual(10, globalNamespace['foo'])
        self.assertEqual(21, localNamespace['foo'])

    def test_execfileUniversalNewlines(self):
        """
        L{execfile} reads in the specified file using universal newlines so
        that scripts written on one platform will work on another.
        """
        for lineEnding in ('\n', '\r', '\r\n'):
            script = self.writeScript("foo = 'okay'" + lineEnding)
            globalNamespace = {'foo': None}
            execfile(script.path, globalNamespace)
            self.assertEqual('okay', globalNamespace['foo'])