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
def _getNextChunk(self, chunks):
    end = 0
    for chunk in chunks:
        if end == 'eof':
            return
        if end != chunk[0]:
            i = chunks.index(chunk)
            chunks.insert(i, (end, chunk[0]))
            return (end, chunk[0] - end)
        end = chunk[1]
    bufSize = int(self.client.transport.conn.options['buffersize'])
    chunks.append((end, end + bufSize))
    return (end, bufSize)