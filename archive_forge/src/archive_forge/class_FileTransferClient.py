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
class FileTransferClient(FileTransferBase):

    def __init__(self, extData={}):
        """
        @param extData: a dict of extended_name : extended_data items
        to be sent to the server.
        """
        FileTransferBase.__init__(self)
        self.extData = {}
        self.counter = 0
        self.openRequests = {}

    def connectionMade(self):
        data = struct.pack('!L', max(self.versions))
        for k, v in self.extData.values():
            data += NS(k) + NS(v)
        self.sendPacket(FXP_INIT, data)

    def connectionLost(self, reason):
        """
        Called when connection to the remote subsystem was lost.

        Any pending requests are aborted.
        """
        FileTransferBase.connectionLost(self, reason)
        if self.openRequests:
            requestError = error.ConnectionLost()
            requestError.__cause__ = reason.value
            requestFailure = failure.Failure(requestError)
            while self.openRequests:
                _, deferred = self.openRequests.popitem()
                deferred.errback(requestFailure)

    def _sendRequest(self, msg, data):
        """
        Send a request and return a deferred which waits for the result.

        @type msg: L{int}
        @param msg: The request type (e.g., C{FXP_READ}).

        @type data: L{bytes}
        @param data: The body of the request.
        """
        if not self.connected:
            return defer.fail(error.ConnectionLost())
        data = struct.pack('!L', self.counter) + data
        d = defer.Deferred()
        self.openRequests[self.counter] = d
        self.counter += 1
        self.sendPacket(msg, data)
        return d

    def _parseRequest(self, data):
        id, = struct.unpack('!L', data[:4])
        d = self.openRequests[id]
        del self.openRequests[id]
        return (d, data[4:])

    def openFile(self, filename, flags, attrs):
        """
        Open a file.

        This method returns a L{Deferred} that is called back with an object
        that provides the L{ISFTPFile} interface.

        @type filename: L{bytes}
        @param filename: a string representing the file to open.

        @param flags: an integer of the flags to open the file with, ORed together.
        The flags and their values are listed at the bottom of this file.

        @param attrs: a list of attributes to open the file with.  It is a
        dictionary, consisting of 0 or more keys.  The possible keys are::

            size: the size of the file in bytes
            uid: the user ID of the file as an integer
            gid: the group ID of the file as an integer
            permissions: the permissions of the file with as an integer.
            the bit representation of this field is defined by POSIX.
            atime: the access time of the file as seconds since the epoch.
            mtime: the modification time of the file as seconds since the epoch.
            ext_*: extended attributes.  The server is not required to
            understand this, but it may.

        NOTE: there is no way to indicate text or binary files.  it is up
        to the SFTP client to deal with this.
        """
        data = NS(filename) + struct.pack('!L', flags) + self._packAttributes(attrs)
        d = self._sendRequest(FXP_OPEN, data)
        d.addCallback(self._cbOpenHandle, ClientFile, filename)
        return d

    def _cbOpenHandle(self, handle, handleClass, name):
        """
        Callback invoked when an OPEN or OPENDIR request succeeds.

        @param handle: The handle returned by the server
        @type handle: L{bytes}
        @param handleClass: The class that will represent the
        newly-opened file or directory to the user (either L{ClientFile} or
        L{ClientDirectory}).
        @param name: The name of the file or directory represented
        by C{handle}.
        @type name: L{bytes}
        """
        cb = handleClass(self, handle)
        cb.name = name
        return cb

    def removeFile(self, filename):
        """
        Remove the given file.

        This method returns a Deferred that is called back when it succeeds.

        @type filename: L{bytes}
        @param filename: the name of the file as a string.
        """
        return self._sendRequest(FXP_REMOVE, NS(filename))

    def renameFile(self, oldpath, newpath):
        """
        Rename the given file.

        This method returns a Deferred that is called back when it succeeds.

        @type oldpath: L{bytes}
        @param oldpath: the current location of the file.
        @type newpath: L{bytes}
        @param newpath: the new file name.
        """
        return self._sendRequest(FXP_RENAME, NS(oldpath) + NS(newpath))

    def makeDirectory(self, path, attrs):
        """
        Make a directory.

        This method returns a Deferred that is called back when it is
        created.

        @type path: L{bytes}
        @param path: the name of the directory to create as a string.

        @param attrs: a dictionary of attributes to create the directory
        with.  Its meaning is the same as the attrs in the openFile method.
        """
        return self._sendRequest(FXP_MKDIR, NS(path) + self._packAttributes(attrs))

    def removeDirectory(self, path):
        """
        Remove a directory (non-recursively)

        It is an error to remove a directory that has files or directories in
        it.

        This method returns a Deferred that is called back when it is removed.

        @type path: L{bytes}
        @param path: the directory to remove.
        """
        return self._sendRequest(FXP_RMDIR, NS(path))

    def openDirectory(self, path):
        """
        Open a directory for scanning.

        This method returns a Deferred that is called back with an iterable
        object that has a close() method.

        The close() method is called when the client is finished reading
        from the directory.  At this point, the iterable will no longer
        be used.

        The iterable returns triples of the form (filename, longname, attrs)
        or a Deferred that returns the same.  The sequence must support
        __getitem__, but otherwise may be any 'sequence-like' object.

        filename is the name of the file relative to the directory.
        logname is an expanded format of the filename.  The recommended format
        is:
        -rwxr-xr-x   1 mjos     staff      348911 Mar 25 14:29 t-filexfer
        1234567890 123 12345678 12345678 12345678 123456789012

        The first line is sample output, the second is the length of the field.
        The fields are: permissions, link count, user owner, group owner,
        size in bytes, modification time.

        attrs is a dictionary in the format of the attrs argument to openFile.

        @type path: L{bytes}
        @param path: the directory to open.
        """
        d = self._sendRequest(FXP_OPENDIR, NS(path))
        d.addCallback(self._cbOpenHandle, ClientDirectory, path)
        return d

    def getAttrs(self, path, followLinks=0):
        """
        Return the attributes for the given path.

        This method returns a dictionary in the same format as the attrs
        argument to openFile or a Deferred that is called back with same.

        @type path: L{bytes}
        @param path: the path to return attributes for as a string.
        @param followLinks: a boolean.  if it is True, follow symbolic links
        and return attributes for the real path at the base.  if it is False,
        return attributes for the specified path.
        """
        if followLinks:
            m = FXP_STAT
        else:
            m = FXP_LSTAT
        return self._sendRequest(m, NS(path))

    def setAttrs(self, path, attrs):
        """
        Set the attributes for the path.

        This method returns when the attributes are set or a Deferred that is
        called back when they are.

        @type path: L{bytes}
        @param path: the path to set attributes for as a string.
        @param attrs: a dictionary in the same format as the attrs argument to
        openFile.
        """
        data = NS(path) + self._packAttributes(attrs)
        return self._sendRequest(FXP_SETSTAT, data)

    def readLink(self, path):
        """
        Find the root of a set of symbolic links.

        This method returns the target of the link, or a Deferred that
        returns the same.

        @type path: L{bytes}
        @param path: the path of the symlink to read.
        """
        d = self._sendRequest(FXP_READLINK, NS(path))
        return d.addCallback(self._cbRealPath)

    def makeLink(self, linkPath, targetPath):
        """
        Create a symbolic link.

        This method returns when the link is made, or a Deferred that
        returns the same.

        @type linkPath: L{bytes}
        @param linkPath: the pathname of the symlink as a string
        @type targetPath: L{bytes}
        @param targetPath: the path of the target of the link as a string.
        """
        return self._sendRequest(FXP_SYMLINK, NS(linkPath) + NS(targetPath))

    def realPath(self, path):
        """
        Convert any path to an absolute path.

        This method returns the absolute path as a string, or a Deferred
        that returns the same.

        @type path: L{bytes}
        @param path: the path to convert as a string.
        """
        d = self._sendRequest(FXP_REALPATH, NS(path))
        return d.addCallback(self._cbRealPath)

    def _cbRealPath(self, result):
        name, longname, attrs = result[0]
        name = name.decode('utf-8')
        return name

    def extendedRequest(self, request, data):
        """
        Make an extended request of the server.

        The method returns a Deferred that is called back with
        the result of the extended request.

        @type request: L{bytes}
        @param request: the name of the extended request to make.
        @type data: L{bytes}
        @param data: any other data that goes along with the request.
        """
        return self._sendRequest(FXP_EXTENDED, NS(request) + data)

    def packet_VERSION(self, data):
        version, = struct.unpack('!L', data[:4])
        data = data[4:]
        d = {}
        while data:
            k, data = getNS(data)
            v, data = getNS(data)
            d[k] = v
        self.version = version
        self.gotServerVersion(version, d)

    def packet_STATUS(self, data):
        d, data = self._parseRequest(data)
        code, = struct.unpack('!L', data[:4])
        data = data[4:]
        if len(data) >= 4:
            msg, data = getNS(data)
            if len(data) >= 4:
                lang, data = getNS(data)
            else:
                lang = b''
        else:
            msg = b''
            lang = b''
        if code == FX_OK:
            d.callback((msg, lang))
        elif code == FX_EOF:
            d.errback(EOFError(msg))
        elif code == FX_OP_UNSUPPORTED:
            d.errback(NotImplementedError(msg))
        else:
            d.errback(SFTPError(code, nativeString(msg), lang))

    def packet_HANDLE(self, data):
        d, data = self._parseRequest(data)
        handle, _ = getNS(data)
        d.callback(handle)

    def packet_DATA(self, data):
        d, data = self._parseRequest(data)
        d.callback(getNS(data)[0])

    def packet_NAME(self, data):
        d, data = self._parseRequest(data)
        count, = struct.unpack('!L', data[:4])
        data = data[4:]
        files = []
        for i in range(count):
            filename, data = getNS(data)
            longname, data = getNS(data)
            attrs, data = self._parseAttributes(data)
            files.append((filename, longname, attrs))
        d.callback(files)

    def packet_ATTRS(self, data):
        d, data = self._parseRequest(data)
        d.callback(self._parseAttributes(data)[0])

    def packet_EXTENDED_REPLY(self, data):
        d, data = self._parseRequest(data)
        d.callback(data)

    def gotServerVersion(self, serverVersion, extData):
        """
        Called when the client sends their version info.

        @param serverVersion: an integer representing the version of the SFTP
        protocol they are claiming.
        @param extData: a dictionary of extended_name : extended_data items.
        These items are sent by the client to indicate additional features.
        """