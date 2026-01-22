import paramiko
import queue
import urllib.parse
import requests.adapters
import logging
import os
import signal
import socket
import subprocess
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
import urllib3
import urllib3.connection
class SSHSocket(socket.socket):

    def __init__(self, host):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = None
        self.user = None
        if ':' in self.host:
            self.host, self.port = self.host.split(':')
        if '@' in self.host:
            self.user, self.host = self.host.split('@')
        self.proc = None

    def connect(self, **kwargs):
        args = ['ssh']
        if self.user:
            args = args + ['-l', self.user]
        if self.port:
            args = args + ['-p', self.port]
        args = args + ['--', self.host, 'docker system dial-stdio']
        preexec_func = None
        if not constants.IS_WINDOWS_PLATFORM:

            def f():
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            preexec_func = f
        env = dict(os.environ)
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('SSL_CERT_FILE', None)
        self.proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stdin=subprocess.PIPE, preexec_fn=preexec_func)

    def _write(self, data):
        if not self.proc or self.proc.stdin.closed:
            raise Exception('SSH subprocess not initiated.connect() must be called first.')
        written = self.proc.stdin.write(data)
        self.proc.stdin.flush()
        return written

    def sendall(self, data):
        self._write(data)

    def send(self, data):
        return self._write(data)

    def recv(self, n):
        if not self.proc:
            raise Exception('SSH subprocess not initiated.connect() must be called first.')
        return self.proc.stdout.read(n)

    def makefile(self, mode):
        if not self.proc:
            self.connect()
        self.proc.stdout.channel = self
        return self.proc.stdout

    def close(self):
        if not self.proc or self.proc.stdin.closed:
            return
        self.proc.stdin.write(b'\n\n')
        self.proc.stdin.flush()
        self.proc.terminate()