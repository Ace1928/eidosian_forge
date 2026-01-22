import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class IFTPShellTestsMixin:
    """
    Generic tests for the C{IFTPShell} interface.
    """

    def directoryExists(self, path):
        """
        Test if the directory exists at C{path}.

        @param path: the relative path to check.
        @type path: C{str}.

        @return: C{True} if C{path} exists and is a directory, C{False} if
            it's not the case
        @rtype: C{bool}
        """
        raise NotImplementedError()

    def createDirectory(self, path):
        """
        Create a directory in C{path}.

        @param path: the relative path of the directory to create, with one
            segment.
        @type path: C{str}
        """
        raise NotImplementedError()

    def fileExists(self, path):
        """
        Test if the file exists at C{path}.

        @param path: the relative path to check.
        @type path: C{str}.

        @return: C{True} if C{path} exists and is a file, C{False} if it's not
            the case.
        @rtype: C{bool}
        """
        raise NotImplementedError()

    def createFile(self, path, fileContent=b''):
        """
        Create a file named C{path} with some content.

        @param path: the relative path of the file to create, without
            directory.
        @type path: C{str}

        @param fileContent: the content of the file.
        @type fileContent: C{str}
        """
        raise NotImplementedError()

    def test_createDirectory(self):
        """
        C{directoryExists} should report correctly about directory existence,
        and C{createDirectory} should create a directory detectable by
        C{directoryExists}.
        """
        self.assertFalse(self.directoryExists('bar'))
        self.createDirectory('bar')
        self.assertTrue(self.directoryExists('bar'))

    def test_createFile(self):
        """
        C{fileExists} should report correctly about file existence, and
        C{createFile} should create a file detectable by C{fileExists}.
        """
        self.assertFalse(self.fileExists('file.txt'))
        self.createFile('file.txt')
        self.assertTrue(self.fileExists('file.txt'))

    def test_makeDirectory(self):
        """
        Create a directory and check it ends in the filesystem.
        """
        d = self.shell.makeDirectory(('foo',))

        def cb(result):
            self.assertTrue(self.directoryExists('foo'))
        return d.addCallback(cb)

    def test_makeDirectoryError(self):
        """
        Creating a directory that already exists should fail with a
        C{ftp.FileExistsError}.
        """
        self.createDirectory('foo')
        d = self.shell.makeDirectory(('foo',))
        return self.assertFailure(d, ftp.FileExistsError)

    def test_removeDirectory(self):
        """
        Try to remove a directory and check it's removed from the filesystem.
        """
        self.createDirectory('bar')
        d = self.shell.removeDirectory(('bar',))

        def cb(result):
            self.assertFalse(self.directoryExists('bar'))
        return d.addCallback(cb)

    def test_removeDirectoryOnFile(self):
        """
        removeDirectory should not work in file and fail with a
        C{ftp.IsNotADirectoryError}.
        """
        self.createFile('file.txt')
        d = self.shell.removeDirectory(('file.txt',))
        return self.assertFailure(d, ftp.IsNotADirectoryError)

    def test_removeNotExistingDirectory(self):
        """
        Removing directory that doesn't exist should fail with a
        C{ftp.FileNotFoundError}.
        """
        d = self.shell.removeDirectory(('bar',))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_removeFile(self):
        """
        Try to remove a file and check it's removed from the filesystem.
        """
        self.createFile('file.txt')
        d = self.shell.removeFile(('file.txt',))

        def cb(res):
            self.assertFalse(self.fileExists('file.txt'))
        d.addCallback(cb)
        return d

    def test_removeFileOnDirectory(self):
        """
        removeFile should not work on directory.
        """
        self.createDirectory('ned')
        d = self.shell.removeFile(('ned',))
        return self.assertFailure(d, ftp.IsADirectoryError)

    def test_removeNotExistingFile(self):
        """
        Try to remove a non existent file, and check it raises a
        L{ftp.FileNotFoundError}.
        """
        d = self.shell.removeFile(('foo',))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_list(self):
        """
        Check the output of the list method.
        """
        self.createDirectory('ned')
        self.createFile('file.txt')
        d = self.shell.list(('.',))

        def cb(l):
            l.sort()
            self.assertEqual(l, [('file.txt', []), ('ned', [])])
        return d.addCallback(cb)

    def test_listWithStat(self):
        """
        Check the output of list with asked stats.
        """
        self.createDirectory('ned')
        self.createFile('file.txt')
        d = self.shell.list(('.',), ('size', 'permissions'))

        def cb(l):
            l.sort()
            self.assertEqual(len(l), 2)
            self.assertEqual(l[0][0], 'file.txt')
            self.assertEqual(l[1][0], 'ned')
            self.assertEqual(len(l[0][1]), 2)
            self.assertEqual(len(l[1][1]), 2)
        return d.addCallback(cb)

    def test_listWithInvalidStat(self):
        """
        Querying an invalid stat should result to a C{AttributeError}.
        """
        self.createDirectory('ned')
        d = self.shell.list(('.',), ('size', 'whateverstat'))
        return self.assertFailure(d, AttributeError)

    def test_listFile(self):
        """
        Check the output of the list method on a file.
        """
        self.createFile('file.txt')
        d = self.shell.list(('file.txt',))

        def cb(l):
            l.sort()
            self.assertEqual(l, [('file.txt', [])])
        return d.addCallback(cb)

    def test_listNotExistingDirectory(self):
        """
        list on a directory that doesn't exist should fail with a
        L{ftp.FileNotFoundError}.
        """
        d = self.shell.list(('foo',))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_access(self):
        """
        Try to access a resource.
        """
        self.createDirectory('ned')
        d = self.shell.access(('ned',))
        return d

    def test_accessNotFound(self):
        """
        access should fail on a resource that doesn't exist.
        """
        d = self.shell.access(('foo',))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_openForReading(self):
        """
        Check that openForReading returns an object providing C{ftp.IReadFile}.
        """
        self.createFile('file.txt')
        d = self.shell.openForReading(('file.txt',))

        def cb(res):
            self.assertTrue(ftp.IReadFile.providedBy(res))
        d.addCallback(cb)
        return d

    def test_openForReadingNotFound(self):
        """
        openForReading should fail with a C{ftp.FileNotFoundError} on a file
        that doesn't exist.
        """
        d = self.shell.openForReading(('ned',))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_openForReadingOnDirectory(self):
        """
        openForReading should not work on directory.
        """
        self.createDirectory('ned')
        d = self.shell.openForReading(('ned',))
        return self.assertFailure(d, ftp.IsADirectoryError)

    def test_openForWriting(self):
        """
        Check that openForWriting returns an object providing C{ftp.IWriteFile}.
        """
        d = self.shell.openForWriting(('foo',))

        def cb1(res):
            self.assertTrue(ftp.IWriteFile.providedBy(res))
            return res.receive().addCallback(cb2)

        def cb2(res):
            self.assertTrue(IConsumer.providedBy(res))
        d.addCallback(cb1)
        return d

    def test_openForWritingExistingDirectory(self):
        """
        openForWriting should not be able to open a directory that already
        exists.
        """
        self.createDirectory('ned')
        d = self.shell.openForWriting(('ned',))
        return self.assertFailure(d, ftp.IsADirectoryError)

    def test_openForWritingInNotExistingDirectory(self):
        """
        openForWring should fail with a L{ftp.FileNotFoundError} if you specify
        a file in a directory that doesn't exist.
        """
        self.createDirectory('ned')
        d = self.shell.openForWriting(('ned', 'idonotexist', 'foo'))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_statFile(self):
        """
        Check the output of the stat method on a file.
        """
        fileContent = b'wobble\n'
        self.createFile('file.txt', fileContent)
        d = self.shell.stat(('file.txt',), ('size', 'directory'))

        def cb(res):
            self.assertEqual(res[0], len(fileContent))
            self.assertFalse(res[1])
        d.addCallback(cb)
        return d

    def test_statDirectory(self):
        """
        Check the output of the stat method on a directory.
        """
        self.createDirectory('ned')
        d = self.shell.stat(('ned',), ('size', 'directory'))

        def cb(res):
            self.assertTrue(res[1])
        d.addCallback(cb)
        return d

    def test_statOwnerGroup(self):
        """
        Check the owner and groups stats.
        """
        self.createDirectory('ned')
        d = self.shell.stat(('ned',), ('owner', 'group'))

        def cb(res):
            self.assertEqual(len(res), 2)
        d.addCallback(cb)
        return d

    def test_statHardlinksNotImplemented(self):
        """
        If L{twisted.python.filepath.FilePath.getNumberOfHardLinks} is not
        implemented, the number returned is 0
        """
        pathFunc = self.shell._path

        def raiseNotImplemented():
            raise NotImplementedError

        def notImplementedFilePath(path):
            f = pathFunc(path)
            f.getNumberOfHardLinks = raiseNotImplemented
            return f
        self.shell._path = notImplementedFilePath
        self.createDirectory('ned')
        d = self.shell.stat(('ned',), ('hardlinks',))
        self.assertEqual(self.successResultOf(d), [0])

    def test_statOwnerGroupNotImplemented(self):
        """
        If L{twisted.python.filepath.FilePath.getUserID} or
        L{twisted.python.filepath.FilePath.getGroupID} are not implemented,
        the owner returned is "0" and the group is returned as "0"
        """
        pathFunc = self.shell._path

        def raiseNotImplemented():
            raise NotImplementedError

        def notImplementedFilePath(path):
            f = pathFunc(path)
            f.getUserID = raiseNotImplemented
            f.getGroupID = raiseNotImplemented
            return f
        self.shell._path = notImplementedFilePath
        self.createDirectory('ned')
        d = self.shell.stat(('ned',), ('owner', 'group'))
        self.assertEqual(self.successResultOf(d), ['0', '0'])

    def test_statNotExisting(self):
        """
        stat should fail with L{ftp.FileNotFoundError} on a file that doesn't
        exist.
        """
        d = self.shell.stat(('foo',), ('size', 'directory'))
        return self.assertFailure(d, ftp.FileNotFoundError)

    def test_invalidStat(self):
        """
        Querying an invalid stat should result to a C{AttributeError}.
        """
        self.createDirectory('ned')
        d = self.shell.stat(('ned',), ('size', 'whateverstat'))
        return self.assertFailure(d, AttributeError)

    def test_rename(self):
        """
        Try to rename a directory.
        """
        self.createDirectory('ned')
        d = self.shell.rename(('ned',), ('foo',))

        def cb(res):
            self.assertTrue(self.directoryExists('foo'))
            self.assertFalse(self.directoryExists('ned'))
        return d.addCallback(cb)

    def test_renameNotExisting(self):
        """
        Renaming a directory that doesn't exist should fail with
        L{ftp.FileNotFoundError}.
        """
        d = self.shell.rename(('foo',), ('bar',))
        return self.assertFailure(d, ftp.FileNotFoundError)