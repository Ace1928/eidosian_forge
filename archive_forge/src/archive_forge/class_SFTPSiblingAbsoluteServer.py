import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SFTPSiblingAbsoluteServer(SFTPAbsoluteServer):
    """A test server for sftp transports where only absolute paths will work.

    It does this by serving from a deeply-nested directory that doesn't exist.
    """

    def create_server(self):
        server = super().create_server()
        server._server_homedir = '/dev/noone/runs/tests/here'
        return server