from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
class SetContentTests(BytesTestCase):
    """
    Tests for L{FilePath.setContent}.
    """

    def test_write(self) -> None:
        """
        Contents of the file referred to by a L{FilePath} can be written using
        L{FilePath.setContent}.
        """
        pathString = self.mktemp()
        path = filepath.FilePath(pathString)
        path.setContent(b'hello, world')
        with open(pathString, 'rb') as fObj:
            contents = fObj.read()
        self.assertEqual(b'hello, world', contents)

    def test_fileClosing(self) -> None:
        """
        If writing to the underlying file raises an exception,
        L{FilePath.setContent} raises that exception after closing the file.
        """
        fp = ExplodingFilePath('')
        self.assertRaises(IOError, fp.setContent, b'blah')
        self.assertTrue(fp.fp.closed)

    def test_nameCollision(self) -> None:
        """
        L{FilePath.setContent} will use a different temporary filename on each
        invocation, so that multiple processes, threads, or reentrant
        invocations will not collide with each other.
        """
        fp = TrackingFilePath(self.mktemp())
        fp.setContent(b'alpha')
        fp.setContent(b'beta')
        openedSiblings = fp.openedPaths()
        self.assertEqual(len(openedSiblings), 2)
        self.assertNotEqual(openedSiblings[0], openedSiblings[1])

    def _assertOneOpened(self, fp: TrackingFilePath[AnyStr], extension: str) -> None:
        """
        Assert that the L{TrackingFilePath} C{fp} was used to open one sibling
        with the given extension.

        @param fp: A L{TrackingFilePath} which should have been used to open
            file at a sibling path.
        @type fp: L{TrackingFilePath}

        @param extension: The extension the sibling path is expected to have
            had.
        @type extension: L{str}

        @raise: C{self.failureException} is raised if the extension of the
            opened file is incorrect or if not exactly one file was opened
            using C{fp}.
        """
        opened = fp.openedPaths()
        self.assertEqual(len(opened), 1, 'expected exactly one opened file')
        self.assertTrue(opened[0].asTextMode().basename().endswith(extension), '{!r} does not end with {!r} extension'.format(opened[0].basename(), extension))

    def test_defaultExtension(self) -> None:
        """
        L{FilePath.setContent} creates temporary files with the extension
        I{.new} if no alternate extension value is given.
        """
        fp = TrackingFilePath(self.mktemp())
        fp.setContent(b'hello')
        self._assertOneOpened(fp, '.new')

    def test_customExtension(self) -> None:
        """
        L{FilePath.setContent} creates temporary files with a user-supplied
        extension so that if it is somehow interrupted while writing them the
        file that it leaves behind will be identifiable.
        """
        fp = TrackingFilePath(self.mktemp())
        fp.setContent(b'goodbye', b'-something-else')
        self._assertOneOpened(fp, '-something-else')