import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def wrap_for_latency(self):
    tcs = self.server.test_case_server
    if tcs.add_latency:
        self.request = SocketDelay(self.request, tcs.add_latency)