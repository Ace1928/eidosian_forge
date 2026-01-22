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
class TelnetConnection(object):
    """
    Class that represents one Telnet connection.
    """

    def __init__(self, conn, addr, application, server, encoding):
        assert isinstance(addr, tuple)
        assert isinstance(application, TelnetApplication)
        assert isinstance(server, TelnetServer)
        assert isinstance(encoding, text_type)
        self.conn = conn
        self.addr = addr
        self.application = application
        self.closed = False
        self.handling_command = True
        self.server = server
        self.encoding = encoding
        self.callback = None
        self.size = Size(rows=40, columns=79)
        _initialize_telnet(conn)

        def get_size():
            return self.size
        self.stdout = _ConnectionStdout(conn, encoding=encoding)
        self.vt100_output = Vt100_Output(self.stdout, get_size, write_binary=False)
        self.eventloop = _TelnetEventLoopInterface(server)
        self.set_application(create_prompt_application())
        application.client_connected(self)
        self.handling_command = False
        self.cli._redraw()

    def set_application(self, app, callback=None):
        """
        Set ``CommandLineInterface`` instance for this connection.
        (This can be replaced any time.)

        :param cli: CommandLineInterface instance.
        :param callback: Callable that takes the result of the CLI.
        """
        assert isinstance(app, Application)
        assert callback is None or callable(callback)
        self.cli = CommandLineInterface(application=app, eventloop=self.eventloop, output=self.vt100_output)
        self.callback = callback
        cb = self.cli.create_eventloop_callbacks()
        inputstream = InputStream(cb.feed_key)
        stdin_decoder_cls = getincrementaldecoder(self.encoding)
        stdin_decoder = [stdin_decoder_cls()]
        self.cli._is_running = True

        def data_received(data):
            """ TelnetProtocolParser 'data_received' callback """
            assert isinstance(data, binary_type)
            try:
                result = stdin_decoder[0].decode(data)
                inputstream.feed(result)
            except UnicodeDecodeError:
                stdin_decoder[0] = stdin_decoder_cls()
                return ''

        def size_received(rows, columns):
            """ TelnetProtocolParser 'size_received' callback """
            self.size = Size(rows=rows, columns=columns)
            cb.terminal_size_changed()
        self.parser = TelnetProtocolParser(data_received, size_received)

    def feed(self, data):
        """
        Handler for incoming data. (Called by TelnetServer.)
        """
        assert isinstance(data, binary_type)
        self.parser.feed(data)
        self.cli._redraw()
        if self.cli.is_returning:
            try:
                return_value = self.cli.return_value()
            except (EOFError, KeyboardInterrupt) as e:
                logger.info('%s, closing connection.', type(e).__name__)
                self.close()
                return
            self._handle_command(return_value)

    def _handle_command(self, command):
        """
        Handle command. This will run in a separate thread, in order not
        to block the event loop.
        """
        logger.info('Handle command %r', command)

        def in_executor():
            self.handling_command = True
            try:
                if self.callback is not None:
                    self.callback(self, command)
            finally:
                self.server.call_from_executor(done)

        def done():
            self.handling_command = False
            if not self.closed:
                self.cli.reset()
                self.cli.buffers[DEFAULT_BUFFER].reset()
                self.cli.renderer.request_absolute_cursor_position()
                self.vt100_output.flush()
                self.cli._redraw()
        self.server.run_in_executor(in_executor)

    def erase_screen(self):
        """
        Erase output screen.
        """
        self.vt100_output.erase_screen()
        self.vt100_output.cursor_goto(0, 0)
        self.vt100_output.flush()

    def send(self, data):
        """
        Send text to the client.
        """
        assert isinstance(data, text_type)
        self.stdout.write(data.replace('\n', '\r\n'))
        self.stdout.flush()

    def close(self):
        """
        Close the connection.
        """
        self.application.client_leaving(self)
        self.conn.close()
        self.closed = True