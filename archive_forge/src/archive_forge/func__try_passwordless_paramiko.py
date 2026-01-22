import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def _try_passwordless_paramiko(server, keyfile):
    """Try passwordless login with paramiko."""
    if paramiko is None:
        msg = 'Paramiko unavailable, '
        if sys.platform == 'win32':
            msg += 'Paramiko is required for ssh tunneled connections on Windows.'
        else:
            msg += 'use OpenSSH.'
        raise ImportError(msg)
    username, server, port = _split_server(server)
    client = paramiko.SSHClient()
    known_hosts = os.path.expanduser('~/.ssh/known_hosts')
    try:
        client.load_host_keys(known_hosts)
    except FileNotFoundError:
        pass
    policy_name = os.environ.get('PYZMQ_PARAMIKO_HOST_KEY_POLICY', None)
    if policy_name:
        policy = getattr(paramiko, f'{policy_name}Policy')
        client.set_missing_host_key_policy(policy())
    try:
        client.connect(server, port, username=username, key_filename=keyfile, look_for_keys=True)
    except paramiko.AuthenticationException:
        return False
    else:
        client.close()
        return True