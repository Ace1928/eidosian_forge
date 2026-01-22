from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
class ExSocket(object):
    """
    Extension of socket to handle recv and send of special data
    """

    def __init__(self, sock):
        self.sock = sock

    def recvall(self, nbytes):
        res = []
        nread = 0
        while nread < nbytes:
            chunk = self.sock.recv(min(nbytes - nread, 1024))
            nread += len(chunk)
            res.append(chunk)
        return b''.join(res)

    def recvint(self):
        return struct.unpack('@i', self.recvall(4))[0]

    def sendint(self, n):
        self.sock.sendall(struct.pack('@i', n))

    def sendstr(self, s):
        self.sendint(len(s))
        self.sock.sendall(s.encode())

    def recvstr(self):
        slen = self.recvint()
        return self.recvall(slen).decode()