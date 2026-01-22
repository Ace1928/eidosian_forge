import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
def _realpath(self, path):
    if self.root == '/':
        return self.canonicalize(path)
    return self.root + self.canonicalize(path)