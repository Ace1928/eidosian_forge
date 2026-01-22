import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class TestingSFTPConnectionHandler(socketserver.BaseRequestHandler):

    def setup(self):
        self.wrap_for_latency()
        tcs = self.server.test_case_server
        ptrans = paramiko.Transport(self.request)
        self.paramiko_transport = ptrans
        ptrans.set_log_channel('brz.paramiko.transport')
        ptrans.add_server_key(tcs.get_host_key())
        ptrans.set_subsystem_handler('sftp', paramiko.SFTPServer, StubSFTPServer, root=tcs._root, home=tcs._server_homedir)
        server = tcs._server_interface(tcs)
        ptrans.start_server(None, server)

    def finish(self):
        self.paramiko_transport.join()

    def wrap_for_latency(self):
        tcs = self.server.test_case_server
        if tcs.add_latency:
            self.request = SocketDelay(self.request, tcs.add_latency)