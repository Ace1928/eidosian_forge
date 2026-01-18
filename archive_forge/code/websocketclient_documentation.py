import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
stop

        Stop trapping WINCH signals and restore the previous WINCH handler.
        