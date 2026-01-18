from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import socket as pysocket
from ansible.module_utils.six import PY2
def write_to_socket(sock, data):
    if hasattr(sock, '_send_until_done'):
        return sock._send_until_done(data)
    elif hasattr(sock, 'send'):
        return sock.send(data)
    else:
        return os.write(sock.fileno(), data)