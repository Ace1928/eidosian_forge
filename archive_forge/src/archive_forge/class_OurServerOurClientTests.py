import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
@skipIf(not unix, "can't run on non-posix computers")
class OurServerOurClientTests(SFTPTestBase):

    def setUp(self):
        SFTPTestBase.setUp(self)
        self.avatar = FileTransferTestAvatar(self.testDir)
        self.server = filetransfer.FileTransferServer(avatar=self.avatar)
        clientTransport = loopback.LoopbackRelay(self.server)
        self.client = filetransfer.FileTransferClient()
        self._serverVersion = None
        self._extData = None

        def _(serverVersion, extData):
            self._serverVersion = serverVersion
            self._extData = extData
        self.client.gotServerVersion = _
        serverTransport = loopback.LoopbackRelay(self.client)
        self.client.makeConnection(clientTransport)
        self.server.makeConnection(serverTransport)
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        self._emptyBuffers()

    def _emptyBuffers(self):
        while self.serverTransport.buffer or self.clientTransport.buffer:
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()

    def tearDown(self):
        self.serverTransport.loseConnection()
        self.clientTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()

    def test_serverVersion(self):
        self.assertEqual(self._serverVersion, 3)
        self.assertEqual(self._extData, {b'conchTest': b'ext data'})

    def test_interface_implementation(self):
        """
        It implements the ISFTPServer interface.
        """
        self.assertTrue(filetransfer.ISFTPServer.providedBy(self.server.client), f'ISFTPServer not provided by {self.server.client!r}')

    def test_openedFileClosedWithConnection(self):
        """
        A file opened with C{openFile} is closed when the connection is lost.
        """
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()
        oldClose = os.close
        closed = []

        def close(fd):
            closed.append(fd)
            oldClose(fd)
        self.patch(os, 'close', close)

        def _fileOpened(openFile):
            fd = self.server.openFiles[openFile.handle[4:]].fd
            self.serverTransport.loseConnection()
            self.clientTransport.loseConnection()
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()
            self.assertEqual(self.server.openFiles, {})
            self.assertIn(fd, closed)
        d.addCallback(_fileOpened)
        return d

    def test_openedDirectoryClosedWithConnection(self):
        """
        A directory opened with C{openDirectory} is close when the connection
        is lost.
        """
        d = self.client.openDirectory('')
        self._emptyBuffers()

        def _getFiles(openDir):
            self.serverTransport.loseConnection()
            self.clientTransport.loseConnection()
            self.serverTransport.clearBuffer()
            self.clientTransport.clearBuffer()
            self.assertEqual(self.server.openDirs, {})
        d.addCallback(_getFiles)
        return d

    def test_openFileIO(self):
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _fileOpened(openFile):
            self.assertEqual(openFile, filetransfer.ISFTPFile(openFile))
            d = _readChunk(openFile)
            d.addCallback(_writeChunk, openFile)
            return d

        def _readChunk(openFile):
            d = openFile.readChunk(0, 20)
            self._emptyBuffers()
            d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10)
            return d

        def _writeChunk(_, openFile):
            d = openFile.writeChunk(20, b'c' * 10)
            self._emptyBuffers()
            d.addCallback(_readChunk2, openFile)
            return d

        def _readChunk2(_, openFile):
            d = openFile.readChunk(0, 30)
            self._emptyBuffers()
            d.addCallback(self.assertEqual, b'a' * 10 + b'b' * 10 + b'c' * 10)
            return d
        d.addCallback(_fileOpened)
        return d

    def test_closedFileGetAttrs(self):
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(_, openFile):
            d = openFile.getAttrs()
            self._emptyBuffers()
            return d

        def _err(f):
            self.flushLoggedErrors()
            return f

        def _close(openFile):
            d = openFile.close()
            self._emptyBuffers()
            d.addCallback(_getAttrs, openFile)
            d.addErrback(_err)
            return self.assertFailure(d, filetransfer.SFTPError)
        d.addCallback(_close)
        return d

    def test_openFileAttributes(self):
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(openFile):
            d = openFile.getAttrs()
            self._emptyBuffers()
            d.addCallback(_getAttrs2)
            return d

        def _getAttrs2(attrs1):
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, attrs1)
            return d
        return d.addCallback(_getAttrs)

    def test_openFileSetAttrs(self):
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {})
        self._emptyBuffers()

        def _getAttrs(openFile):
            d = openFile.getAttrs()
            self._emptyBuffers()
            d.addCallback(_setAttrs)
            return d

        def _setAttrs(attrs):
            attrs['atime'] = 0
            d = self.client.setAttrs(b'testfile1', attrs)
            self._emptyBuffers()
            d.addCallback(_getAttrs2)
            d.addCallback(self.assertEqual, attrs)
            return d

        def _getAttrs2(_):
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            return d
        d.addCallback(_getAttrs)
        return d

    def test_openFileExtendedAttributes(self):
        """
        Check that L{filetransfer.FileTransferClient.openFile} can send
        extended attributes, that should be extracted server side. By default,
        they are ignored, so we just verify they are correctly parsed.
        """
        savedAttributes = {}
        oldOpenFile = self.server.client.openFile

        def openFile(filename, flags, attrs):
            savedAttributes.update(attrs)
            return oldOpenFile(filename, flags, attrs)
        self.server.client.openFile = openFile
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ | filetransfer.FXF_WRITE, {'ext_foo': b'bar'})
        self._emptyBuffers()

        def check(ign):
            self.assertEqual(savedAttributes, {'ext_foo': b'bar'})
        return d.addCallback(check)

    def test_removeFile(self):
        d = self.client.getAttrs(b'testRemoveFile')
        self._emptyBuffers()

        def _removeFile(ignored):
            d = self.client.removeFile(b'testRemoveFile')
            self._emptyBuffers()
            return d
        d.addCallback(_removeFile)
        d.addCallback(_removeFile)
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_renameFile(self):
        d = self.client.getAttrs(b'testRenameFile')
        self._emptyBuffers()

        def _rename(attrs):
            d = self.client.renameFile(b'testRenameFile', b'testRenamedFile')
            self._emptyBuffers()
            d.addCallback(_testRenamed, attrs)
            return d

        def _testRenamed(_, attrs):
            d = self.client.getAttrs(b'testRenamedFile')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, attrs)
        return d.addCallback(_rename)

    def test_directoryBad(self):
        d = self.client.getAttrs(b'testMakeDirectory')
        self._emptyBuffers()
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_directoryCreation(self):
        d = self.client.makeDirectory(b'testMakeDirectory', {})
        self._emptyBuffers()

        def _getAttrs(_):
            d = self.client.getAttrs(b'testMakeDirectory')
            self._emptyBuffers()
            return d

        def _removeDirectory(_):
            d = self.client.removeDirectory(b'testMakeDirectory')
            self._emptyBuffers()
            return d
        d.addCallback(_getAttrs)
        d.addCallback(_removeDirectory)
        d.addCallback(_getAttrs)
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_openDirectory(self):
        d = self.client.openDirectory(b'')
        self._emptyBuffers()
        files = []

        def _getFiles(openDir):

            def append(f):
                files.append(f)
                return openDir
            d = defer.maybeDeferred(openDir.next)
            self._emptyBuffers()
            d.addCallback(append)
            d.addCallback(_getFiles)
            d.addErrback(_close, openDir)
            return d

        def _checkFiles(ignored):
            fs = list(list(zip(*files))[0])
            fs.sort()
            self.assertEqual(fs, [b'.testHiddenFile', b'testDirectory', b'testRemoveFile', b'testRenameFile', b'testfile1'])

        def _close(_, openDir):
            d = openDir.close()
            self._emptyBuffers()
            return d
        d.addCallback(_getFiles)
        d.addCallback(_checkFiles)
        return d

    def test_linkDoesntExist(self):
        d = self.client.getAttrs(b'testLink')
        self._emptyBuffers()
        return self.assertFailure(d, filetransfer.SFTPError)

    def test_linkSharesAttrs(self):
        d = self.client.makeLink(b'testLink', b'testfile1')
        self._emptyBuffers()

        def _getFirstAttrs(_):
            d = self.client.getAttrs(b'testLink', 1)
            self._emptyBuffers()
            return d

        def _getSecondAttrs(firstAttrs):
            d = self.client.getAttrs(b'testfile1')
            self._emptyBuffers()
            d.addCallback(self.assertEqual, firstAttrs)
            return d
        d.addCallback(_getFirstAttrs)
        return d.addCallback(_getSecondAttrs)

    def test_linkPath(self):
        d = self.client.makeLink(b'testLink', b'testfile1')
        self._emptyBuffers()

        def _readLink(_):
            d = self.client.readLink(b'testLink')
            self._emptyBuffers()
            testFile = FilePath(os.getcwd()).preauthChild(self.testDir.path)
            testFile = testFile.child('testfile1')
            d.addCallback(self.assertEqual, testFile.path)
            return d

        def _realPath(_):
            d = self.client.realPath(b'testLink')
            self._emptyBuffers()
            testLink = FilePath(os.getcwd()).preauthChild(self.testDir.path)
            testLink = testLink.child('testfile1')
            d.addCallback(self.assertEqual, testLink.path)
            return d
        d.addCallback(_readLink)
        d.addCallback(_realPath)
        return d

    def test_extendedRequest(self):
        d = self.client.extendedRequest(b'testExtendedRequest', b'foo')
        self._emptyBuffers()
        d.addCallback(self.assertEqual, b'bar')
        d.addCallback(self._cbTestExtendedRequest)
        return d

    def _cbTestExtendedRequest(self, ignored):
        d = self.client.extendedRequest(b'testBadRequest', b'')
        self._emptyBuffers()
        return self.assertFailure(d, NotImplementedError)

    @defer.inlineCallbacks
    def test_openDirectoryIteratorDeprecated(self):
        """
        Using client.openDirectory as an iterator is deprecated.
        """
        d = self.client.openDirectory(b'')
        self._emptyBuffers()
        openDir = (yield d)
        oneFile = openDir.next()
        self._emptyBuffers()
        yield oneFile
        warnings = self.flushWarnings()
        message = 'Using twisted.conch.ssh.filetransfer.ClientDirectory as an iterator was deprecated in Twisted 18.9.0.'
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    @defer.inlineCallbacks
    def test_closedConnectionCancelsRequests(self):
        """
        If there are requests outstanding when the connection
        is closed for any reason, they should fail.
        """
        d = self.client.openFile(b'testfile1', filetransfer.FXF_READ, {})
        self._emptyBuffers()
        fh = (yield d)
        gotReadRequest = []

        def _slowRead(offset, length):
            self.assertEqual(gotReadRequest, [])
            d = defer.Deferred()
            gotReadRequest.append(offset)
            return d
        [serverSideFh] = self.server.openFiles.values()
        serverSideFh.readChunk = _slowRead
        del serverSideFh
        d = fh.readChunk(100, 200)
        self._emptyBuffers()
        self.assertEqual(len(gotReadRequest), 1)
        self.assertNoResult(d)
        self.serverTransport.loseConnection()
        self.serverTransport.clearBuffer()
        self.clientTransport.clearBuffer()
        self._emptyBuffers()
        self.assertFalse(self.client.connected)
        self.failureResultOf(d, ConnectionLost)
        d = fh.getAttrs()
        self.failureResultOf(d, ConnectionLost)