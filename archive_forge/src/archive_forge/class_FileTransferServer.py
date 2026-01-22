import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
class FileTransferServer(FileTransferBase):

    def __init__(self, data=None, avatar=None):
        FileTransferBase.__init__(self)
        self.client = ISFTPServer(avatar)
        self.openFiles = {}
        self.openDirs = {}

    def packet_INIT(self, data):
        version, = struct.unpack('!L', data[:4])
        self.version = min(list(self.versions) + [version])
        data = data[4:]
        ext = {}
        while data:
            extName, data = getNS(data)
            extData, data = getNS(data)
            ext[extName] = extData
        ourExt = self.client.gotVersion(version, ext)
        ourExtData = b''
        for k, v in ourExt.items():
            ourExtData += NS(k) + NS(v)
        self.sendPacket(FXP_VERSION, struct.pack('!L', self.version) + ourExtData)

    def packet_OPEN(self, data):
        requestId = data[:4]
        data = data[4:]
        filename, data = getNS(data)
        flags, = struct.unpack('!L', data[:4])
        data = data[4:]
        attrs, data = self._parseAttributes(data)
        assert data == b'', f'still have data in OPEN: {data!r}'
        d = defer.maybeDeferred(self.client.openFile, filename, flags, attrs)
        d.addCallback(self._cbOpenFile, requestId)
        d.addErrback(self._ebStatus, requestId, b'open failed')

    def _cbOpenFile(self, fileObj, requestId):
        fileId = networkString(str(hash(fileObj)))
        if fileId in self.openFiles:
            raise KeyError('id already open')
        self.openFiles[fileId] = fileObj
        self.sendPacket(FXP_HANDLE, requestId + NS(fileId))

    def packet_CLOSE(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        self._log.info('closing: {requestId!r} {handle!r}', requestId=requestId, handle=handle)
        assert data == b'', f'still have data in CLOSE: {data!r}'
        if handle in self.openFiles:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.close)
            d.addCallback(self._cbClose, handle, requestId)
            d.addErrback(self._ebStatus, requestId, b'close failed')
        elif handle in self.openDirs:
            dirObj = self.openDirs[handle][0]
            d = defer.maybeDeferred(dirObj.close)
            d.addCallback(self._cbClose, handle, requestId, 1)
            d.addErrback(self._ebStatus, requestId, b'close failed')
        else:
            code = errno.ENOENT
            text = os.strerror(code)
            err = OSError(code, text)
            self._ebStatus(failure.Failure(err), requestId)

    def _cbClose(self, result, handle, requestId, isDir=0):
        if isDir:
            del self.openDirs[handle]
        else:
            del self.openFiles[handle]
        self._sendStatus(requestId, FX_OK, b'file closed')

    def packet_READ(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        (offset, length), data = (struct.unpack('!QL', data[:12]), data[12:])
        assert data == b'', f'still have data in READ: {data!r}'
        if handle not in self.openFiles:
            self._ebRead(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.readChunk, offset, length)
            d.addCallback(self._cbRead, requestId)
            d.addErrback(self._ebStatus, requestId, b'read failed')

    def _cbRead(self, result, requestId):
        if result == b'':
            raise EOFError()
        self.sendPacket(FXP_DATA, requestId + NS(result))

    def packet_WRITE(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        offset, = struct.unpack('!Q', data[:8])
        data = data[8:]
        writeData, data = getNS(data)
        assert data == b'', f'still have data in WRITE: {data!r}'
        if handle not in self.openFiles:
            self._ebWrite(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.writeChunk, offset, writeData)
            d.addCallback(self._cbStatus, requestId, b'write succeeded')
            d.addErrback(self._ebStatus, requestId, b'write failed')

    def packet_REMOVE(self, data):
        requestId = data[:4]
        data = data[4:]
        filename, data = getNS(data)
        assert data == b'', f'still have data in REMOVE: {data!r}'
        d = defer.maybeDeferred(self.client.removeFile, filename)
        d.addCallback(self._cbStatus, requestId, b'remove succeeded')
        d.addErrback(self._ebStatus, requestId, b'remove failed')

    def packet_RENAME(self, data):
        requestId = data[:4]
        data = data[4:]
        oldPath, data = getNS(data)
        newPath, data = getNS(data)
        assert data == b'', f'still have data in RENAME: {data!r}'
        d = defer.maybeDeferred(self.client.renameFile, oldPath, newPath)
        d.addCallback(self._cbStatus, requestId, b'rename succeeded')
        d.addErrback(self._ebStatus, requestId, b'rename failed')

    def packet_MKDIR(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        attrs, data = self._parseAttributes(data)
        assert data == b'', f'still have data in MKDIR: {data!r}'
        d = defer.maybeDeferred(self.client.makeDirectory, path, attrs)
        d.addCallback(self._cbStatus, requestId, b'mkdir succeeded')
        d.addErrback(self._ebStatus, requestId, b'mkdir failed')

    def packet_RMDIR(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        assert data == b'', f'still have data in RMDIR: {data!r}'
        d = defer.maybeDeferred(self.client.removeDirectory, path)
        d.addCallback(self._cbStatus, requestId, b'rmdir succeeded')
        d.addErrback(self._ebStatus, requestId, b'rmdir failed')

    def packet_OPENDIR(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        assert data == b'', f'still have data in OPENDIR: {data!r}'
        d = defer.maybeDeferred(self.client.openDirectory, path)
        d.addCallback(self._cbOpenDirectory, requestId)
        d.addErrback(self._ebStatus, requestId, b'opendir failed')

    def _cbOpenDirectory(self, dirObj, requestId):
        handle = networkString(str(hash(dirObj)))
        if handle in self.openDirs:
            raise KeyError('already opened this directory')
        self.openDirs[handle] = [dirObj, iter(dirObj)]
        self.sendPacket(FXP_HANDLE, requestId + NS(handle))

    def packet_READDIR(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        assert data == b'', f'still have data in READDIR: {data!r}'
        if handle not in self.openDirs:
            self._ebStatus(failure.Failure(KeyError()), requestId)
        else:
            dirObj, dirIter = self.openDirs[handle]
            d = defer.maybeDeferred(self._scanDirectory, dirIter, [])
            d.addCallback(self._cbSendDirectory, requestId)
            d.addErrback(self._ebStatus, requestId, b'scan directory failed')

    def _scanDirectory(self, dirIter, f):
        while len(f) < 250:
            try:
                info = next(dirIter)
            except StopIteration:
                if not f:
                    raise EOFError
                return f
            if isinstance(info, defer.Deferred):
                info.addCallback(self._cbScanDirectory, dirIter, f)
                return
            else:
                f.append(info)
        return f

    def _cbScanDirectory(self, result, dirIter, f):
        f.append(result)
        return self._scanDirectory(dirIter, f)

    def _cbSendDirectory(self, result, requestId):
        data = b''
        for filename, longname, attrs in result:
            data += NS(filename)
            data += NS(longname)
            data += self._packAttributes(attrs)
        self.sendPacket(FXP_NAME, requestId + struct.pack('!L', len(result)) + data)

    def packet_STAT(self, data, followLinks=1):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        assert data == b'', f'still have data in STAT/LSTAT: {data!r}'
        d = defer.maybeDeferred(self.client.getAttrs, path, followLinks)
        d.addCallback(self._cbStat, requestId)
        d.addErrback(self._ebStatus, requestId, b'stat/lstat failed')

    def packet_LSTAT(self, data):
        self.packet_STAT(data, 0)

    def packet_FSTAT(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        assert data == b'', f'still have data in FSTAT: {data!r}'
        if handle not in self.openFiles:
            self._ebStatus(failure.Failure(KeyError(f'{handle} not in self.openFiles')), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.getAttrs)
            d.addCallback(self._cbStat, requestId)
            d.addErrback(self._ebStatus, requestId, b'fstat failed')

    def _cbStat(self, result, requestId):
        data = requestId + self._packAttributes(result)
        self.sendPacket(FXP_ATTRS, data)

    def packet_SETSTAT(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        attrs, data = self._parseAttributes(data)
        if data != b'':
            self._log.warn('Still have data in SETSTAT: {data!r}', data=data)
        d = defer.maybeDeferred(self.client.setAttrs, path, attrs)
        d.addCallback(self._cbStatus, requestId, b'setstat succeeded')
        d.addErrback(self._ebStatus, requestId, b'setstat failed')

    def packet_FSETSTAT(self, data):
        requestId = data[:4]
        data = data[4:]
        handle, data = getNS(data)
        attrs, data = self._parseAttributes(data)
        assert data == b'', f'still have data in FSETSTAT: {data!r}'
        if handle not in self.openFiles:
            self._ebStatus(failure.Failure(KeyError()), requestId)
        else:
            fileObj = self.openFiles[handle]
            d = defer.maybeDeferred(fileObj.setAttrs, attrs)
            d.addCallback(self._cbStatus, requestId, b'fsetstat succeeded')
            d.addErrback(self._ebStatus, requestId, b'fsetstat failed')

    def packet_READLINK(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        assert data == b'', f'still have data in READLINK: {data!r}'
        d = defer.maybeDeferred(self.client.readLink, path)
        d.addCallback(self._cbReadLink, requestId)
        d.addErrback(self._ebStatus, requestId, b'readlink failed')

    def _cbReadLink(self, result, requestId):
        self._cbSendDirectory([(result, b'', {})], requestId)

    def packet_SYMLINK(self, data):
        requestId = data[:4]
        data = data[4:]
        linkPath, data = getNS(data)
        targetPath, data = getNS(data)
        d = defer.maybeDeferred(self.client.makeLink, linkPath, targetPath)
        d.addCallback(self._cbStatus, requestId, b'symlink succeeded')
        d.addErrback(self._ebStatus, requestId, b'symlink failed')

    def packet_REALPATH(self, data):
        requestId = data[:4]
        data = data[4:]
        path, data = getNS(data)
        assert data == b'', f'still have data in REALPATH: {data!r}'
        d = defer.maybeDeferred(self.client.realPath, path)
        d.addCallback(self._cbReadLink, requestId)
        d.addErrback(self._ebStatus, requestId, b'realpath failed')

    def packet_EXTENDED(self, data):
        requestId = data[:4]
        data = data[4:]
        extName, extData = getNS(data)
        d = defer.maybeDeferred(self.client.extendedRequest, extName, extData)
        d.addCallback(self._cbExtended, requestId)
        d.addErrback(self._ebStatus, requestId, b'extended ' + extName + b' failed')

    def _cbExtended(self, data, requestId):
        self.sendPacket(FXP_EXTENDED_REPLY, requestId + data)

    def _cbStatus(self, result, requestId, msg=b'request succeeded'):
        self._sendStatus(requestId, FX_OK, msg)

    def _ebStatus(self, reason, requestId, msg=b'request failed'):
        code = FX_FAILURE
        message = msg
        if isinstance(reason.value, (IOError, OSError)):
            if reason.value.errno == errno.ENOENT:
                code = FX_NO_SUCH_FILE
                message = networkString(reason.value.strerror)
            elif reason.value.errno == errno.EACCES:
                code = FX_PERMISSION_DENIED
                message = networkString(reason.value.strerror)
            elif reason.value.errno == errno.EEXIST:
                code = FX_FILE_ALREADY_EXISTS
            else:
                self._log.failure('Request {requestId} failed: {message}', failure=reason, requestId=requestId, message=message)
        elif isinstance(reason.value, EOFError):
            code = FX_EOF
            if reason.value.args:
                message = networkString(reason.value.args[0])
        elif isinstance(reason.value, NotImplementedError):
            code = FX_OP_UNSUPPORTED
            if reason.value.args:
                message = networkString(reason.value.args[0])
        elif isinstance(reason.value, SFTPError):
            code = reason.value.code
            message = networkString(reason.value.message)
        else:
            self._log.failure('Request {requestId} failed with unknown error: {message}', failure=reason, requestId=requestId, message=message)
        self._sendStatus(requestId, code, message)

    def _sendStatus(self, requestId, code, message, lang=b''):
        """
        Helper method to send a FXP_STATUS message.
        """
        data = requestId + struct.pack('!L', code)
        data += NS(message)
        data += NS(lang)
        self.sendPacket(FXP_STATUS, data)

    def connectionLost(self, reason):
        """
        Called when connection to the remote subsystem was lost.

        Clean all opened files and directories.
        """
        FileTransferBase.connectionLost(self, reason)
        for fileObj in self.openFiles.values():
            fileObj.close()
        self.openFiles = {}
        for dirObj, dirIter in self.openDirs.values():
            dirObj.close()
        self.openDirs = {}