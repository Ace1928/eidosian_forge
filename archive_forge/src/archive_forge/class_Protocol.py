import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
class Protocol(ProcessProtocol):
    STATES = SEND_INPUT, COLLECT = range(2)

    def __init__(self):
        self.data = ''
        self.delayed_call = None
        self.states = iter(self.STATES)
        self.state = next(self.states)

    def outReceived(self, data):
        self.data += data.decode(encoding)
        if self.delayed_call is not None:
            self.delayed_call.cancel()
        self.delayed_call = reactor.callLater(0.5, self.next)

    def next(self):
        self.delayed_call = None
        if self.state == self.SEND_INPUT:
            index = self.data.find('>>> ')
            if index >= 0:
                self.data = self.data[index + 4:]
                self.transport.write(input.encode(encoding))
                self.state = next(self.states)
            elif self.data == '\x1b[6n':
                self.transport.write('\x1b[2;1R'.encode(encoding))
        else:
            self.transport.closeStdin()
            if self.transport.pid is not None:
                self.delayed_call = None
                self.transport.signalProcess('TERM')

    def processExited(self, reason):
        if self.delayed_call is not None:
            self.delayed_call.cancel()
        result.callback(self.data)