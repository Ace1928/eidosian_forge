from __future__ import annotations
import fcntl
import grp
import os
import pty
import pwd
import socket
import struct
import time
import tty
from typing import Callable, Dict, Tuple
from zope.interface import implementer
from twisted.conch import ttymodes
from twisted.conch.avatar import ConchUser
from twisted.conch.error import ConchError
from twisted.conch.interfaces import ISession, ISFTPFile, ISFTPServer
from twisted.conch.ls import lsLine
from twisted.conch.ssh import filetransfer, forwarding, session
from twisted.conch.ssh.filetransfer import (
from twisted.cred import portal
from twisted.cred.error import LoginDenied
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.interfaces import IListeningPort
from twisted.logger import Logger
from twisted.python import components
from twisted.python.compat import nativeString
def addUTMPEntry(self, loggedIn=1):
    if not utmp:
        return
    ipAddress = self.avatar.conn.transport.transport.getPeer().host
    packedIp, = struct.unpack('L', socket.inet_aton(ipAddress))
    ttyName = self.ptyTuple[2][5:]
    t = time.time()
    t1 = int(t)
    t2 = int((t - t1) * 1000000.0)
    entry = utmp.UtmpEntry()
    entry.ut_type = loggedIn and utmp.USER_PROCESS or utmp.DEAD_PROCESS
    entry.ut_pid = self.pty.pid
    entry.ut_line = ttyName
    entry.ut_id = ttyName[-4:]
    entry.ut_tv = (t1, t2)
    if loggedIn:
        entry.ut_user = self.avatar.username
        entry.ut_host = socket.gethostbyaddr(ipAddress)[0]
        entry.ut_addr_v6 = (packedIp, 0, 0, 0)
    a = utmp.UtmpRecord(utmp.UTMP_FILE)
    a.pututline(entry)
    a.endutent()
    b = utmp.UtmpRecord(utmp.WTMP_FILE)
    b.pututline(entry)
    b.endutent()