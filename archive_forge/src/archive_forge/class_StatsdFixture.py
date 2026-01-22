import itertools
import os
import pprint
import select
import socket
import threading
import time
import fixtures
from keystoneauth1 import exceptions
import prometheus_client
from requests import exceptions as rexceptions
import testtools.content
from openstack.tests.unit import base
class StatsdFixture(fixtures.Fixture):

    def _setUp(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 0))
        self.port = self.sock.getsockname()[1]
        self.wake_read, self.wake_write = os.pipe()
        self.stats = []
        self.thread.start()
        self.addCleanup(self._cleanup)

    def run(self):
        while self.running:
            poll = select.poll()
            poll.register(self.sock, select.POLLIN)
            poll.register(self.wake_read, select.POLLIN)
            ret = poll.poll()
            for fd, event in ret:
                if fd == self.sock.fileno():
                    data = self.sock.recvfrom(1024)
                    if not data:
                        return
                    self.stats.append(data[0])
                if fd == self.wake_read:
                    return

    def _cleanup(self):
        self.running = False
        os.write(self.wake_write, b'1\n')
        self.thread.join()