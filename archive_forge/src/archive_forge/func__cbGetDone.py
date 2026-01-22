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
def _cbGetDone(self, ignored, rf, lf):
    log.msg('get done')
    rf.close()
    lf.close()
    if self.useProgressBar:
        self._writeToTransport('\n')
    return f'Transferred {rf.name} to {lf.name}'