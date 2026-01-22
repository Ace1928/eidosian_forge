import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
class TestThreadedServer(SocketThread):

    def __init__(self, test, sock, prog, timeout, max_clients):
        threading.Thread.__init__(self, None, None, 'test-server')
        self.daemon = True
        self._clients = 0
        self._finished_clients = 0
        self._max_clients = max_clients
        self._timeout = timeout
        self._sock = sock
        self._active = True
        self._prog = prog
        self._s1, self._s2 = socket.socketpair()
        self._s1.setblocking(False)
        self._test = test

    def stop(self):
        try:
            if self._s2 and self._s2.fileno() != -1:
                try:
                    self._s2.send(b'stop')
                except OSError:
                    pass
        finally:
            super().stop()

    def run(self):
        try:
            with self._sock:
                self._sock.setblocking(0)
                self._run()
        finally:
            self._s1.close()
            self._s2.close()

    def _run(self):
        while self._active:
            if self._clients >= self._max_clients:
                return
            r, w, x = select.select([self._sock, self._s1], [], [], self._timeout)
            if self._s1 in r:
                return
            if self._sock in r:
                try:
                    conn, addr = self._sock.accept()
                except BlockingIOError:
                    continue
                except socket.timeout:
                    if not self._active:
                        return
                    else:
                        raise
                else:
                    self._clients += 1
                    conn.settimeout(self._timeout)
                    try:
                        with conn:
                            self._handle_client(conn)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as ex:
                        self._active = False
                        try:
                            raise
                        finally:
                            self._test._abort_socket_test(ex)

    def _handle_client(self, sock):
        self._prog(TestSocketWrapper(sock))

    @property
    def addr(self):
        return self._sock.getsockname()