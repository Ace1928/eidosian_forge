from __future__ import unicode_literals
import socket
import select
import threading
import os
import fcntl
from six import int2byte, text_type, binary_type
from codecs import getincrementaldecoder
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.eventloop.base import EventLoop
from prompt_toolkit.interface import CommandLineInterface, Application
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.shortcuts import create_prompt_application
from prompt_toolkit.terminal.vt100_input import InputStream
from prompt_toolkit.terminal.vt100_output import Vt100_Output
from .log import logger
from .protocol import IAC, DO, LINEMODE, SB, MODE, SE, WILL, ECHO, NAWS, SUPPRESS_GO_AHEAD
from .protocol import TelnetProtocolParser
from .application import TelnetApplication
class TelnetServer(object):
    """
    Telnet server implementation.
    """

    def __init__(self, host='127.0.0.1', port=23, application=None, encoding='utf-8'):
        assert isinstance(host, text_type)
        assert isinstance(port, int)
        assert isinstance(application, TelnetApplication)
        assert isinstance(encoding, text_type)
        self.host = host
        self.port = port
        self.application = application
        self.encoding = encoding
        self.connections = set()
        self._calls_from_executor = []
        self._schedule_pipe = os.pipe()
        fcntl.fcntl(self._schedule_pipe[0], fcntl.F_SETFL, os.O_NONBLOCK)

    @classmethod
    def create_socket(cls, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(4)
        return s

    def run_in_executor(self, callback):
        threading.Thread(target=callback).start()

    def call_from_executor(self, callback):
        self._calls_from_executor.append(callback)
        if self._schedule_pipe:
            os.write(self._schedule_pipe[1], b'x')

    def _process_callbacks(self):
        """
        Process callbacks from `call_from_executor` in eventloop.
        """
        os.read(self._schedule_pipe[0], 1024)
        calls_from_executor, self._calls_from_executor = (self._calls_from_executor, [])
        for c in calls_from_executor:
            c()

    def run(self):
        """
        Run the eventloop for the telnet server.
        """
        listen_socket = self.create_socket(self.host, self.port)
        logger.info('Listening for telnet connections on %s port %r', self.host, self.port)
        try:
            while True:
                self.connections = set([c for c in self.connections if not c.closed])
                connections = set([c for c in self.connections if not c.handling_command])
                read_list = [listen_socket, self._schedule_pipe[0]] + [c.conn for c in connections]
                read, _, _ = select.select(read_list, [], [])
                for s in read:
                    if s == listen_socket:
                        self._accept(listen_socket)
                    elif s == self._schedule_pipe[0]:
                        self._process_callbacks()
                    else:
                        self._handle_incoming_data(s)
        finally:
            listen_socket.close()

    def _accept(self, listen_socket):
        """
        Accept new incoming connection.
        """
        conn, addr = listen_socket.accept()
        connection = TelnetConnection(conn, addr, self.application, self, encoding=self.encoding)
        self.connections.add(connection)
        logger.info('New connection %r %r', *addr)

    def _handle_incoming_data(self, conn):
        """
        Handle incoming data on socket.
        """
        connection = [c for c in self.connections if c.conn == conn][0]
        data = conn.recv(1024)
        if data:
            connection.feed(data)
        else:
            self.connections.remove(connection)