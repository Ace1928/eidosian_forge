import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def try_passwordless_ssh(server, keyfile, paramiko=None):
    """Attempt to make an ssh connection without a password.
    This is mainly used for requiring password input only once
    when many tunnels may be connected to the same server.

    If paramiko is None, the default for the platform is chosen.
    """
    if paramiko is None:
        paramiko = sys.platform == 'win32'
    if not paramiko:
        f = _try_passwordless_openssh
    else:
        f = _try_passwordless_paramiko
    return f(server, keyfile)