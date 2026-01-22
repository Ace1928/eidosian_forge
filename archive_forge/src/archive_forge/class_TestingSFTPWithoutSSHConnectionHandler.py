import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class TestingSFTPWithoutSSHConnectionHandler(TestingSFTPConnectionHandler):

    def setup(self):
        self.wrap_for_latency()

        class FakeChannel:

            def get_transport(self):
                return self

            def get_log_channel(self):
                return 'brz.paramiko'

            def get_name(self):
                return '1'

            def get_hexdump(self):
                return False

            def close(self):
                pass
        tcs = self.server.test_case_server
        sftp_server = paramiko.SFTPServer(FakeChannel(), 'sftp', StubServer(tcs), StubSFTPServer, root=tcs._root, home=tcs._server_homedir)
        self.sftp_server = sftp_server
        sys_stderr = sys.stderr
        try:
            sftp_server.start_subsystem('sftp', None, ssh.SocketAsChannelAdapter(self.request))
        except OSError as e:
            if len(e.args) > 0 and e.args[0] == errno.EPIPE:
                pass
            else:
                raise
        except Exception as e:
            sys_stderr.write('\nEXCEPTION {!r}: '.format(e.__class__))
            sys_stderr.write('{}\n\n'.format(e))

    def finish(self):
        self.sftp_server.finish_subsystem()