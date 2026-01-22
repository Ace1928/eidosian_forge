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
class _TelnetEventLoopInterface(EventLoop):
    """
    Eventloop object to be assigned to `CommandLineInterface`.
    """

    def __init__(self, server):
        self._server = server

    def close(self):
        """ Ignore. """

    def stop(self):
        """ Ignore. """

    def run_in_executor(self, callback):
        self._server.run_in_executor(callback)

    def call_from_executor(self, callback, _max_postpone_until=None):
        self._server.call_from_executor(callback)

    def add_reader(self, fd, callback):
        raise NotImplementedError

    def remove_reader(self, fd):
        raise NotImplementedError