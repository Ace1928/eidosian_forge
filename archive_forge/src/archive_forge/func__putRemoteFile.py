import fcntl
import fnmatch
import getpass
import glob
import os
import pwd
import stat
import struct
import sys
import tty
from typing import List, Optional, TextIO, Union
from twisted.conch.client import connect, default, options
from twisted.conch.ssh import channel, common, connection, filetransfer
from twisted.internet import defer, reactor, stdio, utils
from twisted.protocols import basic
from twisted.python import failure, log, usage
from twisted.python.filepath import FilePath
def _putRemoteFile(self, localStream, remotePath):
    """
        Do an upload request.

        @param localStream: Local stream from where data is read.
        @type localStream: File like object.

        @param remotePath: Remote path for the request relative to current working directory.
        @type remotePath: L{str}

        @return: A deferred which fires when transfer is done.
        """
    remote = os.path.join(self.currentDirectory, remotePath)
    flags = filetransfer.FXF_WRITE | filetransfer.FXF_CREAT | filetransfer.FXF_TRUNC
    d = self.client.openFile(remote, flags, {})
    d.addCallback(self._cbPutOpenFile, localStream)
    d.addErrback(self._ebCloseLf, localStream)
    return d