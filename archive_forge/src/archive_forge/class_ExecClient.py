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
class ExecClient(WebSocketClient):

    def __init__(self, zunclient, url, exec_id, id, escape='~', close_wait=0.5):
        super(ExecClient, self).__init__(zunclient, url, id, escape, close_wait)
        self.exec_id = exec_id

    def tty_resize(self, height, width):
        """Resize the tty session

        Get the client and send the tty size data to zun api server
        The environment variables need to get when implement sending
        operation.
        """
        height = str(height)
        width = str(width)
        self.cs.containers.execute_resize(self.id, self.exec_id, width, height)