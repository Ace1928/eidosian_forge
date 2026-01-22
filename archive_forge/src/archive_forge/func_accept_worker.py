import socket
import sys
import threading
from debugpy.common import log
from debugpy.common.util import hide_thread_from_debugger
def accept_worker():
    while True:
        try:
            sock, (other_host, other_port) = listener.accept()
        except (OSError, socket.error):
            break
        log.info('Accepted incoming {0} connection from {1}:{2}.', name, other_host, other_port)
        handler(sock)