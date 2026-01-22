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
class UnicodeFilePathTests(TestCase):
    """
    L{FilePath} instances should have the same internal representation as they
    were instantiated with.
    """

    def test_UnicodeInstantiation(self) -> None:
        """
        L{FilePath} instantiated with a text path will return a text-mode
        FilePath.
        """
        fp = filepath.FilePath('./mon€y')
        self.assertEqual(type(fp.path), str)

    def test_UnicodeInstantiationBytesChild(self) -> None:
        """
        Calling L{FilePath.child} on a text-mode L{FilePath} with a L{bytes}
        subpath will return a bytes-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y')
        child = fp.child('child-mon€y'.encode())
        self.assertEqual(type(child.path), bytes)

    def test_UnicodeInstantiationUnicodeChild(self) -> None:
        """
        Calling L{FilePath.child} on a text-mode L{FilePath} with a text
        subpath will return a text-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y')
        child = fp.child('mon€y')
        self.assertEqual(type(child.path), str)

    def test_UnicodeInstantiationUnicodePreauthChild(self) -> None:
        """
        Calling L{FilePath.preauthChild} on a text-mode L{FilePath} with a text
        subpath will return a text-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y')
        child = fp.preauthChild('mon€y')
        self.assertEqual(type(child.path), str)

    def test_UnicodeInstantiationBytesPreauthChild(self) -> None:
        """
        Calling L{FilePath.preauthChild} on a text-mode L{FilePath} with a bytes
        subpath will return a bytes-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y')
        child = fp.preauthChild('child-mon€y'.encode())
        self.assertEqual(type(child.path), bytes)

    def test_BytesInstantiation(self) -> None:
        """
        L{FilePath} instantiated with a L{bytes} path will return a bytes-mode
        FilePath.
        """
        fp = filepath.FilePath(b'./')
        self.assertEqual(type(fp.path), bytes)

    def test_BytesInstantiationBytesChild(self) -> None:
        """
        Calling L{FilePath.child} on a bytes-mode L{FilePath} with a bytes
        subpath will return a bytes-mode FilePath.
        """
        fp = filepath.FilePath(b'./')
        child = fp.child('child-mon€y'.encode())
        self.assertEqual(type(child.path), bytes)

    def test_BytesInstantiationUnicodeChild(self) -> None:
        """
        Calling L{FilePath.child} on a bytes-mode L{FilePath} with a text
        subpath will return a text-mode FilePath.
        """
        fp = filepath.FilePath('parent-mon€y'.encode())
        child = fp.child('mon€y')
        self.assertEqual(type(child.path), str)

    def test_BytesInstantiationBytesPreauthChild(self) -> None:
        """
        Calling L{FilePath.preauthChild} on a bytes-mode L{FilePath} with a
        bytes subpath will return a bytes-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y'.encode())
        child = fp.preauthChild('child-mon€y'.encode())
        self.assertEqual(type(child.path), bytes)

    def test_BytesInstantiationUnicodePreauthChild(self) -> None:
        """
        Calling L{FilePath.preauthChild} on a bytes-mode L{FilePath} with a text
        subpath will return a text-mode FilePath.
        """
        fp = filepath.FilePath('./parent-mon€y'.encode())
        child = fp.preauthChild('mon€y')
        self.assertEqual(type(child.path), str)

    @skipIf(platform.isWindows(), 'Test will not work on Windows')
    def test_unicoderepr(self) -> None:
        """
        The repr of a L{unicode} L{FilePath} shouldn't burst into flames.
        """
        fp = filepath.FilePath('/mon€y')
        reprOutput = repr(fp)
        self.assertEqual("FilePath('/mon€y')", reprOutput)

    @skipIf(platform.isWindows(), 'Test will not work on Windows')
    def test_bytesrepr(self) -> None:
        """
        The repr of a L{bytes} L{FilePath} shouldn't burst into flames.
        """
        fp = filepath.FilePath('/parent-mon€y'.encode())
        reprOutput = repr(fp)
        self.assertEqual("FilePath(b'/parent-mon\\xe2\\x82\\xacy')", reprOutput)

    @skipIf(not platform.isWindows(), 'Test only works on Windows')
    def test_unicodereprWindows(self) -> None:
        """
        The repr of a L{unicode} L{FilePath} shouldn't burst into flames.
        """
        fp = filepath.FilePath('C:\\')
        reprOutput = repr(fp)
        self.assertEqual("FilePath('C:\\\\')", reprOutput)

    @skipIf(not platform.isWindows(), 'Test only works on Windows')
    def test_bytesreprWindows(self) -> None:
        """
        The repr of a L{bytes} L{FilePath} shouldn't burst into flames.
        """
        fp = filepath.FilePath(b'C:\\')
        reprOutput = repr(fp)
        self.assertEqual("FilePath(b'C:\\\\')", reprOutput)

    def test_mixedTypeGlobChildren(self) -> None:
        """
        C{globChildren} will return the same type as the pattern argument.
        """
        fp = filepath.FilePath('/')
        children = fp.globChildren(b'*')
        self.assertIsInstance(children[0].path, bytes)

    def test_unicodeGlobChildren(self) -> None:
        """
        C{globChildren} works with L{unicode}.
        """
        fp = filepath.FilePath('/')
        children = fp.globChildren('*')
        self.assertIsInstance(children[0].path, str)

    def test_unicodeBasename(self) -> None:
        """
        Calling C{basename} on an text- L{FilePath} returns L{unicode}.
        """
        fp = filepath.FilePath('./')
        self.assertIsInstance(fp.basename(), str)

    def test_unicodeDirname(self) -> None:
        """
        Calling C{dirname} on a text-mode L{FilePath} returns L{unicode}.
        """
        fp = filepath.FilePath('./')
        self.assertIsInstance(fp.dirname(), str)

    def test_unicodeParent(self) -> None:
        """
        Calling C{parent} on a text-mode L{FilePath} will return a text-mode
        L{FilePath}.
        """
        fp = filepath.FilePath('./')
        parent = fp.parent()
        self.assertIsInstance(parent.path, str)

    def test_mixedTypeTemporarySibling(self) -> None:
        """
        A L{bytes} extension to C{temporarySibling} will mean a L{bytes} mode
        L{FilePath} is returned.
        """
        fp = filepath.FilePath('./mon€y')
        tempSibling = fp.temporarySibling(b'.txt')
        self.assertIsInstance(tempSibling.path, bytes)

    def test_unicodeTemporarySibling(self) -> None:
        """
        A L{unicode} extension to C{temporarySibling} will mean a L{unicode}
        mode L{FilePath} is returned.
        """
        fp = filepath.FilePath('/tmp/mon€y')
        tempSibling = fp.temporarySibling('.txt')
        self.assertIsInstance(tempSibling.path, str)

    def test_mixedTypeSiblingExtensionSearch(self) -> None:
        """
        C{siblingExtensionSearch} called with L{bytes} on a L{unicode}-mode
        L{FilePath} will return a L{list} of L{bytes}-mode L{FilePath}s.
        """
        fp = filepath.FilePath('./mon€y')
        sibling = filepath.FilePath(fp._asTextPath() + '.txt')
        sibling.touch()
        newPath = fp.siblingExtensionSearch(b'.txt')
        assert newPath is not None
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, bytes)

    def test_unicodeSiblingExtensionSearch(self) -> None:
        """
        C{siblingExtensionSearch} called with L{unicode} on a L{unicode}-mode
        L{FilePath} will return a L{list} of L{unicode}-mode L{FilePath}s.
        """
        fp = filepath.FilePath('./mon€y')
        sibling = filepath.FilePath(fp._asTextPath() + '.txt')
        sibling.touch()
        newPath = fp.siblingExtensionSearch('.txt')
        assert newPath is not None
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, str)

    def test_mixedTypeSiblingExtension(self) -> None:
        """
        C{siblingExtension} called with L{bytes} on a L{unicode}-mode
        L{FilePath} will return a L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./mon€y')
        sibling = filepath.FilePath(fp._asTextPath() + '.txt')
        sibling.touch()
        newPath = fp.siblingExtension(b'.txt')
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, bytes)

    def test_unicodeSiblingExtension(self) -> None:
        """
        C{siblingExtension} called with L{unicode} on a L{unicode}-mode
        L{FilePath} will return a L{unicode}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./mon€y')
        sibling = filepath.FilePath(fp._asTextPath() + '.txt')
        sibling.touch()
        newPath = fp.siblingExtension('.txt')
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, str)

    def test_selfSiblingExtensionSearch(self) -> None:
        """
        C{siblingExtension} passed an empty string should return the same path,
        in the type of its argument.
        """
        exists = filepath.FilePath(self.mktemp())
        exists.touch()
        notExists = filepath.FilePath(self.mktemp())
        self.assertEqual(exists.siblingExtensionSearch(b''), exists.asBytesMode())
        self.assertEqual(exists.siblingExtensionSearch(''), exists.asTextMode())
        self.assertEqual(notExists.siblingExtensionSearch(''), None)
        self.assertEqual(notExists.siblingExtensionSearch(b''), None)

    def test_mixedTypeChildSearchPreauth(self) -> None:
        """
        C{childSearchPreauth} called with L{bytes} on a L{unicode}-mode
        L{FilePath} will return a L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./mon€y')
        fp.createDirectory()
        self.addCleanup(lambda: fp.remove())
        child = fp.child('text.txt')
        child.touch()
        newPath = fp.childSearchPreauth(b'text.txt')
        assert newPath is not None
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, bytes)

    def test_unicodeChildSearchPreauth(self) -> None:
        """
        C{childSearchPreauth} called with L{unicode} on a L{unicode}-mode
        L{FilePath} will return a L{unicode}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./mon€y')
        fp.createDirectory()
        self.addCleanup(lambda: fp.remove())
        child = fp.child('text.txt')
        child.touch()
        newPath = fp.childSearchPreauth('text.txt')
        assert newPath is not None
        self.assertIsInstance(newPath, filepath.FilePath)
        self.assertIsInstance(newPath.path, str)

    def test_asBytesModeFromUnicode(self) -> None:
        """
        C{asBytesMode} on a L{unicode}-mode L{FilePath} returns a new
        L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./tmp')
        newfp = fp.asBytesMode()
        self.assertIsNot(fp, newfp)
        self.assertIsInstance(newfp.path, bytes)

    def test_asTextModeFromBytes(self) -> None:
        """
        C{asBytesMode} on a L{unicode}-mode L{FilePath} returns a new
        L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath(b'./tmp')
        newfp = fp.asTextMode()
        self.assertIsNot(fp, newfp)
        self.assertIsInstance(newfp.path, str)

    def test_asBytesModeFromBytes(self) -> None:
        """
        C{asBytesMode} on a L{bytes}-mode L{FilePath} returns the same
        L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath(b'./tmp')
        newfp = fp.asBytesMode()
        self.assertIs(fp, newfp)
        self.assertIsInstance(newfp.path, bytes)

    def test_asTextModeFromUnicode(self) -> None:
        """
        C{asTextMode} on a L{unicode}-mode L{FilePath} returns the same
        L{unicode}-mode L{FilePath}.
        """
        fp = filepath.FilePath('./tmp')
        newfp = fp.asTextMode()
        self.assertIs(fp, newfp)
        self.assertIsInstance(newfp.path, str)

    def test_asBytesModeFromUnicodeWithEncoding(self) -> None:
        """
        C{asBytesMode} with an C{encoding} argument uses that encoding when
        coercing the L{unicode}-mode L{FilePath} to a L{bytes}-mode L{FilePath}.
        """
        fp = filepath.FilePath('☃')
        newfp = fp.asBytesMode(encoding='utf-8')
        self.assertIn(b'\xe2\x98\x83', newfp.path)

    def test_asTextModeFromBytesWithEncoding(self) -> None:
        """
        C{asTextMode} with an C{encoding} argument uses that encoding when
        coercing the L{bytes}-mode L{FilePath} to a L{unicode}-mode L{FilePath}.
        """
        fp = filepath.FilePath(b'\xe2\x98\x83')
        newfp = fp.asTextMode(encoding='utf-8')
        self.assertIn('☃', newfp.path)

    def test_asBytesModeFromUnicodeWithUnusableEncoding(self) -> None:
        """
        C{asBytesMode} with an C{encoding} argument that can't be used to encode
        the unicode path raises a L{UnicodeError}.
        """
        fp = filepath.FilePath('☃')
        with self.assertRaises(UnicodeError):
            fp.asBytesMode(encoding='ascii')

    def test_asTextModeFromBytesWithUnusableEncoding(self) -> None:
        """
        C{asTextMode} with an C{encoding} argument that can't be used to encode
        the unicode path raises a L{UnicodeError}.
        """
        fp = filepath.FilePath(b'\\u2603')
        with self.assertRaises(UnicodeError):
            fp.asTextMode(encoding='utf-32')