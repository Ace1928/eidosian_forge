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
def _initialize_telnet(connection):
    logger.info('Initializing telnet connection')
    connection.send(IAC + DO + LINEMODE)
    connection.send(IAC + WILL + SUPPRESS_GO_AHEAD)
    connection.send(IAC + SB + LINEMODE + MODE + int2byte(0) + IAC + SE)
    connection.send(IAC + WILL + ECHO)
    connection.send(IAC + DO + NAWS)