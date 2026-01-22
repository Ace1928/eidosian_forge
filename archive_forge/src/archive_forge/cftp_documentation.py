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

        Parse line received as command line input and return first filename
        together with the remaining line.

        @param line: Arguments received from command line input.
        @type line: L{str}

        @return: Tupple with filename and rest. Return empty values when no path was not found.
        @rtype: C{tupple}
        