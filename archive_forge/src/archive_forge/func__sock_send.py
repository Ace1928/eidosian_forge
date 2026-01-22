import logging
import socket
from re import sub
from gunicorn.glogging import Logger
def _sock_send(self, msg):
    try:
        if isinstance(msg, str):
            msg = msg.encode('ascii')
        if self.dogstatsd_tags:
            msg = msg + b'|#' + self.dogstatsd_tags.encode('ascii')
        if self.sock:
            self.sock.send(msg)
    except Exception:
        Logger.warning(self, 'Error sending message to statsd', exc_info=True)