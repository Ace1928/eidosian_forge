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
def _cbGetPrivateKey(self, ans, count):
    file = os.path.expanduser(self.usedFiles[-1])
    try:
        return keys.Key.fromFile(file, password=ans).keyObject
    except keys.BadKeyError:
        if count == 2:
            raise
        prompt = "Enter passphrase for key '%s': " % self.usedFiles[-1]
        return deferredAskFrame(prompt, 0).addCallback(self._cbGetPrivateKey, count + 1)