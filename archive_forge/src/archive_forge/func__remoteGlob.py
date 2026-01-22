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
def _remoteGlob(self, fullPath):
    log.msg('looking up %s' % fullPath)
    head, tail = os.path.split(fullPath)
    if '*' in tail or '?' in tail:
        glob = 1
    else:
        glob = 0
    if tail and (not glob):
        d = self.client.openDirectory(fullPath)
        d.addCallback(self._cbOpenList, '')
        d.addErrback(self._ebNotADirectory, head, tail)
    else:
        d = self.client.openDirectory(head)
        d.addCallback(self._cbOpenList, tail)
    return d