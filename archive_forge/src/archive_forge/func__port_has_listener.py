from __future__ import print_function
from builtins import str
from builtins import range
import io
import os
import platform
import socket
import subprocess
import sys
import time
def _port_has_listener(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0