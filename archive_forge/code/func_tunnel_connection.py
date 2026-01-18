import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def tunnel_connection(socket, addr, server, keyfile=None, password=None, paramiko=None, timeout=60):
    """Connect a socket to an address via an ssh tunnel.

    This is a wrapper for socket.connect(addr), when addr is not accessible
    from the local machine.  It simply creates an ssh tunnel using the remaining args,
    and calls socket.connect('tcp://localhost:lport') where lport is the randomly
    selected local port of the tunnel.

    """
    new_url, tunnel = open_tunnel(addr, server, keyfile=keyfile, password=password, paramiko=paramiko, timeout=timeout)
    socket.connect(new_url)
    return tunnel