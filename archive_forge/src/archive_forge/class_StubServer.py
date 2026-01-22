import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class StubServer(paramiko.ServerInterface):

    def __init__(self, test_case_server):
        paramiko.ServerInterface.__init__(self)
        self.log = test_case_server.log

    def check_auth_password(self, username, password):
        self.log('sftpserver - authorizing: {}'.format(username))
        return paramiko.AUTH_SUCCESSFUL

    def check_channel_request(self, kind, chanid):
        self.log('sftpserver - channel request: {}, {}'.format(kind, chanid))
        return paramiko.OPEN_SUCCEEDED