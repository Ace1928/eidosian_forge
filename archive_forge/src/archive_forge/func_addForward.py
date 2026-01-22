import base64
import getpass
import os
import signal
import struct
import sys
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from typing import List, Tuple
from twisted.conch import error
from twisted.conch.client.default import isInKnownHosts
from twisted.conch.ssh import (
from twisted.conch.ui import tkvt100
from twisted.internet import defer, protocol, reactor, tksupport
from twisted.python import log, usage
def addForward(self):
    port = self.forwardPort.get()
    self.forwardPort.delete(0, Tkinter.END)
    host = self.forwardHost.get()
    self.forwardHost.delete(0, Tkinter.END)
    if self.localRemoteVar.get() == 'local':
        self.forwards.insert(Tkinter.END, f'L:{port}:{host}')
    else:
        self.forwards.insert(Tkinter.END, f'R:{port}:{host}')