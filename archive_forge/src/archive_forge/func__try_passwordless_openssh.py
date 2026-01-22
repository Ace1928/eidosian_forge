import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def _try_passwordless_openssh(server, keyfile):
    """Try passwordless login with shell ssh command."""
    if pexpect is None:
        raise ImportError('pexpect unavailable, use paramiko')
    cmd = 'ssh -f ' + server
    if keyfile:
        cmd += ' -i ' + keyfile
    cmd += ' exit'
    env = os.environ.copy()
    env.pop('SSH_ASKPASS', None)
    ssh_newkey = 'Are you sure you want to continue connecting'
    p = pexpect.spawn(cmd, env=env)
    MAX_RETRY = 10
    for _ in range(MAX_RETRY):
        try:
            i = p.expect([ssh_newkey, _password_pat], timeout=0.1)
            if i == 0:
                raise SSHException("The authenticity of the host can't be established.")
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            return True
        else:
            return False
    raise MaxRetryExceeded(f'Failed after {MAX_RETRY} attempts')